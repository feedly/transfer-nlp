"""
This class contains the abstraction interface to customize runners.
For the training loop, we use the engine logic from pytorch-ignite

Check experiments for examples of experiment json files

"""
import inspect
import logging
from itertools import zip_longest
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Loss, Metric, RunningAverage
from ignite.utils import convert_tensor

from transfer_nlp.loaders.loaders import DatasetSplits
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.regularizers import RegularizerABC

logger = logging.getLogger(__name__)


def set_seed_everywhere(seed: int, cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _prepare_batch(batch: Dict, device=None, non_blocking: bool = False):
    """Prepare batch for training: pass to a device with options.

    """
    result = {key: convert_tensor(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
    return result


class TrainingMetric(Metric):

    def __init__(self, metric: Metric):
        self.source_metric = metric
        self.reset()

        super().__init__(lambda x: x[:-1])

    def reset(self):
        self.source_metric.reset()

    def update(self, output):
        self.source_metric.update(output)

    def compute(self):
        return self.source_metric.compute()


@register_plugin
class BasicTrainer:

    def __init__(self,
                 model: nn.Module,
                 dataset_splits: DatasetSplits,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 metrics: List[Metric],
                 device: str = None,
                 num_epochs: int = 1,
                 seed: int = None,
                 cuda: bool = None,
                 loss_accumulation_steps: int = 4,
                 scheduler: Any = None,  # no common parent class?
                 regularizer: RegularizerABC = None,
                 gradient_clipping: float = 1.0,
                 output_transform=None):

        self.model: nn.Module = model

        self.forward_params = {}

        model_spec = inspect.getfullargspec(model.forward)
        for fparam, pdefault in zip_longest(reversed(model_spec.args[1:]), reversed(model_spec.defaults if model_spec.defaults else [])):
            self.forward_params[fparam] = pdefault

        self.dataset_splits: DatasetSplits = dataset_splits
        self.loss: nn.Module = loss
        self.optimizer: optim.Optimizer = optimizer
        self.metrics: List[Metric] = metrics
        self.device: str = device
        self.num_epochs: int = num_epochs
        self.scheduler: Any = scheduler
        self.seed: int = seed
        self.cuda: bool = cuda
        if self.cuda is None:  # If cuda not specified, just check if the cuda is available and use accordingly
            self.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_accumulation_steps: int = loss_accumulation_steps
        self.regularizer: RegularizerABC = regularizer
        self.gradient_clipping: float = gradient_clipping
        self.output_transform = output_transform

        if self.output_transform:
            self.trainer, training_metrics = self.create_supervised_trainer(output_transform=self.output_transform)
            self.evaluator = self.create_supervised_evaluator(output_transform=self.output_transform)
        else:
            self.trainer, training_metrics = self.create_supervised_trainer()
            self.evaluator = self.create_supervised_evaluator()

        loss_metrics = [m for m in metrics if isinstance(m, Loss)]

        if self.scheduler:
            if not loss_metrics:
                raise ValueError('A loss metric must be configured')
            elif len(loss_metrics) > 1:
                logging.warning('multiple loss metrics detected, using %s for LR scheduling', loss_metrics[0])
            self.loss_metric = loss_metrics[0]

        self.setup(training_metrics)

    def setup(self, training_metrics):
        def metric_name(n) -> str:
            if n.endswith('Accuracy'):
                n = 'acc'
            else:
                n = n[:-6] if n.endswith('Metric') else n
            return n

        def print_metrics(metrics) -> str:
            rv = ''
            metric_keys = sorted(k for k in metrics)
            for k in metric_keys:
                if k == 'Accuracy':
                    rv += f'{metric_name(k)}: {metrics[k]:.3}'
                else:
                    rv += f'{metric_name(k)}: {metrics[k]:.6}'
            return rv

        if self.seed:
            set_seed_everywhere(self.seed, self.cuda)

        pbar = ProgressBar()

        names = []
        for k, v in training_metrics.items():
            name = f'r{k}'
            names.append(name)
            RunningAverage(v).attach(self.trainer, name)
        RunningAverage(None, output_transform=lambda x: x[-1] * self.loss_accumulation_steps).attach(self.trainer, 'rloss')
        names.append('rloss')
        pbar.attach(self.trainer, names)

        pbar = ProgressBar()
        pbar.attach(self.evaluator)

        # A few events handler. To add / modify the events handler, you need to extend the __init__ method of RunnerABC
        # Ignite provides the necessary abstractions and a furnished repository of useful tools

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.evaluator.run(self.dataset_splits.val_data_loader())
            metrics = self.evaluator.state.metrics
            logger.info(f"Validation Results - Epoch: {trainer.state.epoch} {print_metrics(metrics)}")

            if self.scheduler:
                self.scheduler.step(metrics[self.loss_metric.__class__.__name__])

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(trainer):
            self.evaluator.run(self.dataset_splits.test_data_loader())
            metrics = self.evaluator.state.metrics
            logger.info(f"Test Results - Epoch: {trainer.state.epoch} {print_metrics(metrics)}")

    def _forward(self, batch):
        model_inputs = {}
        for p, pdefault in self.forward_params.items():
            val = batch.get(p)
            if val is None:
                if pdefault is None:
                    raise ValueError(f'missing model parameter "{p}"')
                else:
                    val = pdefault

            model_inputs[p] = val

        return self.model(**model_inputs)

    def create_supervised_trainer(self, prepare_batch=_prepare_batch, non_blocking=False,
                                  output_transform=lambda y_pred, y_target, loss: (y_pred, y_target, loss)):

        if self.device:
            self.model.to(self.device)

        # Gradient accumulation trick adapted from :
        # https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        accumulation_steps = self.loss_accumulation_steps

        def _update(engine, batch):

            self.model.train()
            batch = prepare_batch(batch, device=self.device, non_blocking=non_blocking)
            y_pred = self._forward(batch)
            loss = self.loss(input=y_pred, target=batch['y_target'])

            loss /= accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            if engine.state.iteration % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            return output_transform(y_pred, batch['y_target'], loss.item())

        engine = Engine(_update)
        metrics = {}
        for i, metric in enumerate(self.metrics):
            if not isinstance(metric, Loss):
                n = metric.__class__.__name__
                tm = TrainingMetric(metric)
                metrics[n] = tm
                tm.attach(engine, n)

        return engine, metrics

    def create_supervised_evaluator(self, prepare_batch=_prepare_batch, non_blocking=False, output_transform=lambda y, y_pred: (y, y_pred)):

        if self.device:
            self.model.to(self.device)

        def _inference(engine, batch):
            self.model.eval()
            with torch.no_grad():
                batch = prepare_batch(batch, device=self.device, non_blocking=non_blocking)
                y_pred = self._forward(batch)
                return output_transform(y_pred, batch['y_target'])

        engine = Engine(_inference)

        for i, metric in enumerate(self.metrics):
            metric.attach(engine, f'{str(metric.__class__.__name__)}')

        return engine

    def train(self):
        self.trainer.run(self.dataset_splits.train_data_loader(), max_epochs=self.num_epochs)
