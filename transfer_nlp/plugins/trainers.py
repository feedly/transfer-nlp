"""
This class contains the abstraction interface to customize runners.
For the training loop, we use the engine logic from pytorch-ignite

Check experiments for examples of experiment json files

Examples using gradual unfreezing and adaptation methods in general are adapted from
the NAACL 2019 tutorial on TRansfer Learning for NLP https://colab.research.google.com/drive/1iDHCYIrWswIKp-n-pOg69xLoZO09MEgf#scrollTo=GObQkkttljWv&forceEdit=true&offline=true&sandboxMode=true
"""

import inspect
import logging
import re
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, WeightsScalarHandler, WeightsHistHandler, \
    GradsScalarHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Loss, Metric, RunningAverage, MetricsLambda, Accuracy
from ignite.utils import convert_tensor

from transfer_nlp.loaders.loaders import DatasetSplits
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.regularizers import RegularizerABC
from transfer_nlp.plugins.trainer_abc import TrainerABC

logger = logging.getLogger(__name__)

# Tensorboard is used within PyTorch but is not a dependency, so it should be installed manually by users
TENSORBOARD = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    logger.debug("To use torch.utils.tensorboard, please install tensorboard>=1.14, and future")
    TENSORBOARD = False


def set_seed_everywhere(seed: int, cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _prepare_batch(batch: Union[Dict, List, Tuple], device=None, non_blocking: bool = False):
    """Prepare batch for training: pass to a device with options.

    """
    if isinstance(batch, dict):
        result = {key: convert_tensor(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
        return result
    elif isinstance(batch, tuple):
        result = (convert_tensor(value, device=device, non_blocking=non_blocking) for value in batch)
        return result
    elif isinstance(batch, list):
        result = [convert_tensor(value, device=device, non_blocking=non_blocking) for value in batch]
        return result
    else:
        raise ValueError("Only dict, tuples and lists are valid for batch")


class TrainingMetric(Metric):

    def __init__(self, metric: Metric):
        self.source_metric = metric
        self.reset()

        super().__init__(lambda x: x[:-1])

    def reset(self):
        self.source_metric.reset()

    def update(self, output):

        if not isinstance(self.source_metric, MetricsLambda):
            self.source_metric.update(output)
            return

        # If a source metric is made of several metrics, e.g. MetricsLambda
        # metrics, we need to update each sub-metrics separately
        for source in self.source_metric.args:
            if isinstance(source, Metric):
                source.update(output)
        return

    def compute(self):
        return self.source_metric.compute()


@register_plugin
class BaseIgniteTrainer(TrainerABC):

    def __init__(self,
                 model: nn.Module,
                 dataset_splits: DatasetSplits,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 metrics: Dict[str, Metric],
                 device: str = None,
                 num_epochs: int = 1,
                 seed: int = None,
                 cuda: bool = None,
                 loss_accumulation_steps: int = 4,
                 scheduler: Any = None,
                 regularizer: RegularizerABC = None,
                 gradient_clipping: float = 1.0,
                 output_transform=None,
                 tensorboard_logs: str = None):

        self.model: nn.Module = model

        self.forward_param_defaults = {}

        model_spec = inspect.getfullargspec(model.forward)
        self.forward_params: List[str] = model_spec.args[1:]
        for fparam, pdefault in zip(reversed(model_spec.args[1:]), reversed(model_spec.defaults if model_spec.defaults else [])):
            self.forward_param_defaults[fparam] = pdefault

        self.dataset_splits: DatasetSplits = dataset_splits
        self.loss: nn.Module = loss
        self.optimizer: optim.Optimizer = optimizer
        self.metrics: Dict[str, Metric] = metrics
        self.metrics: List[Metric] = [metric for _, metric in self.metrics.items()]
        self.device: str = device
        self.num_epochs: int = num_epochs
        self.scheduler: Any = scheduler
        self.seed: int = seed
        self.cuda: bool = cuda
        if self.cuda is None:  # If cuda not specified, just check if the cuda is available and use accordingly
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_accumulation_steps: int = loss_accumulation_steps
        self.regularizer: RegularizerABC = regularizer
        self.gradient_clipping: float = gradient_clipping
        self.output_transform = output_transform
        self.tensorboard_logs: str = tensorboard_logs
        if self.tensorboard_logs and TENSORBOARD:
            self.writer = SummaryWriter(log_dir=self.tensorboard_logs)

        if not self.output_transform:
            self.output_transform = lambda y_pred, y_target, loss: (y_pred, y_target, loss)

        self.eval_output_transform = lambda y, y_pred: (y, y_pred)

        self.trainer, self.training_metrics = self.create_supervised_trainer()
        self.evaluator = self.create_supervised_evaluator()

        loss_metrics = [m for m in self.metrics if isinstance(m, Loss)]

        if self.scheduler:
            if not loss_metrics:
                raise ValueError('A loss metric must be configured')
            elif len(loss_metrics) > 1:
                logging.warning('multiple loss metrics detected, using %s for LR scheduling', loss_metrics[0])
            self.loss_metric = loss_metrics[0]

        self.metrics_history = {
            "training": defaultdict(list),
            "validation": defaultdict(list),
            "test": defaultdict(list)}

        self.setup(self.training_metrics)

    def setup(self, training_metrics: Dict):
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
                    rv += f'{metric_name(k)}: {metrics[k]:.3} | '
                else:
                    rv += f'{metric_name(k)}: {metrics[k]} | '
            return rv

        def store_metrics(metrics: Dict, mode: str):
            metric_keys = sorted(k for k in metrics)
            for k in metric_keys:
                self.metrics_history[mode][metric_name(k)].append(metrics[k])

        if self.seed:
            set_seed_everywhere(self.seed, self.cuda)

        pbar = ProgressBar(persist=True)

        names = []
        for k, v in training_metrics.items():
            name = f'r{k}'
            names.append(name)
            RunningAverage(v).attach(self.trainer, name)
        RunningAverage(None, output_transform=lambda x: x[-1]).attach(self.trainer, 'rloss')

        names.append('rloss')
        pbar.attach(self.trainer, names)

        ProgressBar(persist=True).attach(engine=self.evaluator, metric_names=names)

        # A few events handler. To add / modify the events handler, you need to extend the __init__ method of RunnerABC
        # Ignite provides the necessary abstractions and a furnished repository of useful tools

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_validation_results(trainer):

            self.evaluator.run(self.dataset_splits.train_data_loader())
            metrics = self.evaluator.state.metrics
            store_metrics(metrics=metrics, mode="training")
            logger.info(f"Training Results - Epoch: {trainer.state.epoch} {print_metrics(metrics)}")

            self.evaluator.run(self.dataset_splits.val_data_loader())
            metrics = self.evaluator.state.metrics
            store_metrics(metrics=metrics, mode="validation")
            logger.info(f"Validation Results - Epoch: {trainer.state.epoch} {print_metrics(metrics)}")

            metrics = self.trainer.state.metrics
            if self.scheduler:
                self.scheduler.step(metrics["rloss"])
                # self.scheduler.step(metrics[self.loss_metric.__class__.__name__])

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(trainer):
            if self.dataset_splits.test_set:
                self.evaluator.run(self.dataset_splits.test_data_loader())
                metrics = self.evaluator.state.metrics
                store_metrics(metrics=metrics, mode="test")
                logger.info(f"Test Results - Epoch: {trainer.state.epoch} {print_metrics(metrics)}")

    def _forward(self, batch):
        model_inputs = {}
        for p in self.forward_params:
            val = batch.get(p)
            if val is None:
                if p in self.forward_param_defaults:
                    val = self.forward_param_defaults[p]
                else:
                    raise ValueError(f'missing model parameter "{p}"')

            model_inputs[p] = val

        return self.model(**model_inputs)

    @abstractmethod
    def update_engine(self, engine, batch):
        raise NotImplementedError

    @abstractmethod
    def infer_engine(self, engine, batch):
        raise NotImplementedError

    def create_supervised_trainer(self):

        if self.device:
            self.model.to(self.device)

        engine = Engine(self.update_engine)
        metrics = {}
        for i, metric in enumerate(self.metrics):
            if not isinstance(metric, Loss):
                n = metric.__class__.__name__
                tm = TrainingMetric(metric)
                metrics[n] = tm
                tm.attach(engine, n)

        return engine, metrics

    def create_supervised_evaluator(self):

        if self.device:
            self.model.to(self.device)

        engine = Engine(self.infer_engine)

        for i, metric in enumerate(self.metrics):
            metric.attach(engine, f'{str(metric.__class__.__name__)}')

        return engine

    def train(self):
        raise NotImplementedError


@register_plugin
class SingleTaskTrainer(BaseIgniteTrainer):

    def __init__(self,
                 model: nn.Module,
                 dataset_splits: DatasetSplits,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 metrics: Dict[str, Metric],
                 device: str = None,
                 num_epochs: int = 1,
                 seed: int = None,
                 cuda: bool = None,
                 loss_accumulation_steps: int = 4,
                 scheduler: Any = None,
                 regularizer: RegularizerABC = None,
                 gradient_clipping: float = 1.0,
                 output_transform=None,
                 tensorboard_logs: str = None,
                 optional_tensorboard_features: bool = False,
                 embeddings_name: str = None):

        super().__init__(
            model=model,
            dataset_splits=dataset_splits,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            num_epochs=num_epochs,
            seed=seed,
            cuda=cuda,
            loss_accumulation_steps=loss_accumulation_steps,
            scheduler=scheduler,
            regularizer=regularizer,
            gradient_clipping=gradient_clipping,
            output_transform=output_transform,
            tensorboard_logs=tensorboard_logs)

        self.optional_tensorboard_features: bool = optional_tensorboard_features
        self.embeddings_name: str = embeddings_name

        self.custom_setup()

    def custom_setup(self):

        if self.tensorboard_logs:
            tb_logger = TensorboardLogger(log_dir=self.tensorboard_logs)
            tb_logger.attach(self.trainer,
                             log_handler=OutputHandler(tag="training", output_transform=lambda loss: {
                                 'loss': loss}),
                             event_name=Events.ITERATION_COMPLETED)
            tb_logger.attach(self.evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["LossMetric"],
                                                       another_engine=self.trainer),
                             event_name=Events.EPOCH_COMPLETED)

            if self.optional_tensorboard_features:
                tb_logger.attach(self.trainer,
                                 log_handler=OptimizerParamsHandler(self.optimizer),
                                 event_name=Events.ITERATION_STARTED)
                tb_logger.attach(self.trainer,
                                 log_handler=WeightsScalarHandler(self.model),
                                 event_name=Events.ITERATION_COMPLETED)
                tb_logger.attach(self.trainer,
                                 log_handler=WeightsHistHandler(self.model),
                                 event_name=Events.EPOCH_COMPLETED)
                tb_logger.attach(self.trainer,
                                 log_handler=GradsScalarHandler(self.model),
                                 event_name=Events.ITERATION_COMPLETED)

            # This is important to close the tensorboard file logger
            @self.trainer.on(Events.COMPLETED)
            def end_tensorboard(trainer):
                logger.info("Training completed")
                tb_logger.close()

        if self.embeddings_name:
            @self.trainer.on(Events.COMPLETED)
            def log_embeddings(trainer):
                if hasattr(self.model, self.embeddings_name) and hasattr(self.dataset_splits, "vectorizer") and TENSORBOARD:
                    logger.info(f"Logging embeddings ({self.embeddings_name}) to Tensorboard!")
                    embeddings = getattr(self.model, self.embeddings_name).weight.data
                    metadata = [str(self.dataset_splits.vectorizer.data_vocab._id2token[token_index]).encode('utf-8') for token_index in
                                range(embeddings.shape[0])]
                    self.writer.add_embedding(mat=embeddings, metadata=metadata, global_step=self.trainer.state.epoch)

    def update_engine(self, engine, batch):

        # Gradient accumulation trick adapted from :
        # https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

        self.model.train()
        batch = _prepare_batch(batch, device=self.device, non_blocking=False)
        if isinstance(batch, dict):
            y_pred = self._forward(batch)
            loss = self.loss(input=y_pred, target=batch['y_target'])
        elif isinstance(batch, tuple) or isinstance(batch, list):
            y_pred = self.model.forward(*batch[:-1])
            loss = self.loss(input=y_pred, target=batch[-1])
        else:
            raise ValueError("Only dict, tuples and lists are valid for batch")

        # Add a regularisation term at train time only
        if self.regularizer:
            loss += self.regularizer.compute_penalty(model=self.model)

        loss /= self.loss_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

        if engine.state.iteration % self.loss_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if isinstance(batch, dict):
            return self.output_transform(y_pred, batch['y_target'], loss.item())
        elif isinstance(batch, tuple) or isinstance(batch, list):
            return self.output_transform(y_pred, batch[-1], loss.item())
        else:
            raise ValueError("Only dict, tuples and lists are valid for batch")

    def infer_engine(self, engine, batch):

        self.model.eval()
        with torch.no_grad():
            batch = _prepare_batch(batch, device=self.device, non_blocking=False)
            if isinstance(batch, dict):
                y_pred = self._forward(batch)
                return self.eval_output_transform(y_pred, batch['y_target'])
            elif isinstance(batch, tuple) or isinstance(batch, list):
                y_pred = self.model.forward(*batch[:-1])
                return self.eval_output_transform(y_pred, batch[-1])
            else:
                raise ValueError("Only dict, tuples and lists are valid for batch")

    def train(self):
        """
        Launch the ignite training pipeline
        If fine-tuning mode is granted in the config file, freeze all layers, replace classification layer by a Linear layer
        and reset the optimizer
        :return:
        """

        self.trainer.run(self.dataset_splits.train_data_loader(), max_epochs=self.num_epochs)


@register_plugin
class SingleTaskFineTuner(SingleTaskTrainer):

    def __init__(self,
                 model: nn.Module,
                 dataset_splits: DatasetSplits,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 metrics: Dict[str, Metric],
                 device: str = None,
                 num_epochs: int = 1,
                 seed: int = None,
                 cuda: bool = None,
                 loss_accumulation_steps: int = 4,
                 scheduler: Any = None,
                 regularizer: RegularizerABC = None,
                 gradient_clipping: float = 1.0,
                 output_transform=None,
                 tensorboard_logs: str = None,
                 optional_tensorboard_features: bool = False,
                 embeddings_name: str = None,
                 adaptation: str = 'hard-freezing',
                 decreasing_factor: int = 2.6,
                 pretrained: bool = False):
        super().__init__(
            model=model,
            dataset_splits=dataset_splits,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            num_epochs=num_epochs,
            seed=seed,
            cuda=cuda,
            loss_accumulation_steps=loss_accumulation_steps,
            scheduler=scheduler,
            regularizer=regularizer,
            gradient_clipping=gradient_clipping,
            output_transform=output_transform,
            tensorboard_logs=tensorboard_logs,
            optional_tensorboard_features=optional_tensorboard_features,
            embeddings_name=embeddings_name
        )
        self.adaptation: str = adaptation
        self.decreasing_factor: int = decreasing_factor
        self.pretrained: bool = pretrained

    def load_pretrained_model(self):
        """
        This methid is not implemented so that pytorch_pretrained_bert is not a 
        required dependency. Use these lines to implement the method if using
        pytorch_pretrained_bert
        Returns:

        """
        # from pytorch_pretrained_bert import cached_path
        # logger.info("Loading pretrained model")
        # state_dict = torch.load(cached_path("https://s3.amazonaws.com/models.huggingface.co/"
        #                                     "naacl-2019-tutorial/model_checkpoint.pth"), map_location=self.device)
        # self.model.load_state_dict(state_dict, strict=False)
        # logger.info("Pretrained model loaded!")

        raise NotImplementedError

    def freeze_params(self):

        for name, param in self.model.named_parameters():
            if 'embeddings' not in name and 'classification' not in name and 'adapters_1' not in name and 'adapters_2' not in name:
                param.detach_()
                param.requires_grad = False

            else:
                param.requires_grad = True

        full_parameters = sum(p.numel() for p in self.model.parameters())
        trained_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"We will train {trained_parameters:3e} parameters out of {full_parameters:3e},"
                    f" i.e. {100 * trained_parameters / full_parameters:.2f}%")

    def gradual_unfreezing(self):

        for name, param in self.model.named_parameters():
            if 'embeddings' not in name and 'classification' not in name:
                param.detach_()
                param.requires_grad = False

            else:
                param.requires_grad = True

        full_parameters = sum(p.numel() for p in self.model.parameters())
        trained_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"We will start by training {trained_parameters:3e} parameters out of {full_parameters:3e},"
                    f" i.e. {100 * trained_parameters / full_parameters:.2f}%")

        # We will unfreeze blocks regularly along the training: one block every `unfreezing_interval` step
        unfreezing_interval = int(len(self.dataset_splits.train_data_loader()) * self.num_epochs / (self.model.num_layers + 1))

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def unfreeze_layer_if_needed(engine):
            if engine.state.iteration % unfreezing_interval == 0:
                # Which layer should we unfreeze now
                unfreezing_index = self.model.num_layers - (engine.state.iteration // unfreezing_interval)

                # Let's unfreeze it
                unfreezed = []
                for name, param in self.model.named_parameters():
                    if re.match(r"transformer\.[^\.]*\." + str(unfreezing_index) + r"\.", name):
                        unfreezed.append(name)
                        param.require_grad = True
                logger.info(f"Unfreezing block {unfreezing_index} with {unfreezed}")

    def discriminative_learning(self):

        logger.info("Using discriminative learning as adaptation strategy")
        # Build parameters groups by layer, numbered from the top ['1', '2', ..., '15']
        parameter_groups = []
        for i in range(self.model.num_layers):
            name_pattern = r"transformer\.[^\.]*\." + str(i) + r"\."
            group = {
                'name': str(self.model.num_layers - i),
                'params': [p for n, p in self.model.named_parameters() if re.match(name_pattern, n)]}
            parameter_groups.append(group)

        # Add the rest of the parameters (embeddings and classification layer) in a group labeled '0'
        name_pattern = r"transformer\.[^\.]*\.\d*\."
        group = {
            'name': '0',
            'params': [p for n, p in self.model.named_parameters() if not re.match(name_pattern, n)]}
        parameter_groups.append(group)

        # Sanity check that we still have the same number of parameters
        assert sum(p.numel() for g in parameter_groups for p in g['params']) \
               == sum(p.numel() for p in self.model.parameters())

        @self.trainer.on(Events.ITERATION_STARTED)
        def update_layer_learning_rates(engine):
            for param_group in self.optimizer.param_groups:
                layer_index = int(param_group["name"])
                param_group["lr"] = param_group["lr"] / (self.decreasing_factor ** layer_index)

        return parameter_groups

    def train(self):

        # if self.pretrained:
        #     self.load_pretrained_model()
        #
        # if self.adaptation == 'hard-freezing':
        #     self.freeze_params()
        # elif self.adaptation == 'gradual-unfreezing':
        #     self.gradual_unfreezing()
        # elif self.adaptation == 'discriminative-learning':
        #     parameter_groups = self.discriminative_learning()
        #     self.experiment_config.factories['optimizer'].kwargs['params'] = parameter_groups
        #     self.experiment_config.factories['optimizer'].param2config_key['params'] = parameter_groups
        # else:
        #     raise ValueError("Transfer NLP supports only hard freezing, gradual unfreezing and discriminative learning")
        #
        # self.optimizer = self.experiment_config.factories['optimizer'].create()
        # self.trainer.run(self.dataset_splits.train_data_loader(), max_epochs=self.num_epochs)
        raise NotImplementedError


@register_plugin
class MultiTaskTrainer(BaseIgniteTrainer):

    def __init__(self,
                 model: nn.Module,
                 dataset_splits: DatasetSplits,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 metrics: Dict[str, Metric],
                 device: str = None,
                 num_epochs: int = 1,
                 seed: int = None,
                 cuda: bool = None,
                 loss_accumulation_steps: int = 4,
                 scheduler: Any = None,
                 regularizer: RegularizerABC = None,
                 gradient_clipping: float = 1.0,
                 output_transform=None,
                 tensorboard_logs: str = None,
                 clf_loss_coef: float = 0.1,
                 lm_loss_coef: float = 0.9
                 ):

        super().__init__(
            model=model,
            dataset_splits=dataset_splits,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            num_epochs=num_epochs,
            seed=seed,
            cuda=cuda,
            loss_accumulation_steps=loss_accumulation_steps,
            scheduler=scheduler,
            regularizer=regularizer,
            gradient_clipping=gradient_clipping,
            output_transform=output_transform,
            tensorboard_logs=tensorboard_logs)
        self.clf_loss_coef = clf_loss_coef
        self.lm_loss_coef = lm_loss_coef
        RunningAverage(Accuracy(output_transform=lambda x: (x[0], x[1]))).attach(self.trainer, 'acc')

    def update_engine(self, engine, batch):
        self.model.train()
        batch = _prepare_batch(batch, device=self.device, non_blocking=False)
        lm_logits, clf_logits = self._forward(batch)
        loss_lm, loss_clf = self.loss(lm_logits=lm_logits, clf_logits=clf_logits, lm_labels=batch['x'], clf_labels=batch['y_target'])
        loss = (self.clf_loss_coef * loss_clf
                + self.lm_loss_coef * loss_lm) / self.loss_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

        if engine.state.iteration % self.loss_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return clf_logits, batch['y_target'], loss.item()

    def infer_engine(self, engine, batch):

        self.model.eval()
        with torch.no_grad():
            batch = _prepare_batch(batch, device=self.device, non_blocking=False)
            lm_logits, clf_logits = self._forward(batch)
            return clf_logits, batch['y_target']

    def create_supervised_evaluator(self):

        if self.device:
            self.model.to(self.device)

        engine = Engine(self.infer_engine)

        Accuracy().attach(engine, "accuracy")

        return engine

    def train(self):
        self.trainer.run(self.dataset_splits.train_data_loader(), max_epochs=self.num_epochs)
