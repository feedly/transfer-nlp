"""
This class contains the abstraction interface to customize runners.
For the training loop, we temporarily let the choice between 2 alternatives, both customizable:

- Defining the training engine explicitely, including all events handling
- Defining the training engine using pytorch-ignite

Check transfer_nlp.experiments for examples of experiment json files

This class also provide useful methods to freeze / unfreeze components of a model.
We will use them in examples of transfer learning
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import convert_tensor
from smart_open import open
from tensorboardX import SummaryWriter
from tqdm import tqdm

from transfer_nlp.embeddings.embeddings import make_embedding_matrix
from transfer_nlp.loaders.loaders import CustomDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.registry import Scheduler, LossFunction, Model, Optimizer, Data, Generator, Metrics, Regularizer
from transfer_nlp.runners.utils import set_seed_everywhere, handle_dirs, make_training_state, update_train_state

name = 'transfer_nlp.runners.runnersABC'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


class RunnerABC:

    def __init__(self, config_args: Dict):

        self.config_args = config_args
        self.dataset_cls = Data(config_args=self.config_args).dataset
        self.dataset: CustomDataset = None
        self.training_state: Dict = {}
        self.vectorizer: Vectorizer = None
        self.loss_func: nn.modules.loss._Loss = None
        self.optimizer: optim.optimizer.Optimizer = None
        self.scheduler: Scheduler = None
        self.mask_index: int = None
        self.epoch_index: int = 0
        self.writer = SummaryWriter(log_dir=self.config_args['logs'])
        self.loss: LossFunction = None
        self.model: nn.Module = None
        self.generator: Generator = None
        self.metrics: Metrics = None
        if self.config_args.get('Regularizer'):
            self.regularizer: Regularizer = None
        else:
            logger.info("Are you sure you don't want to use a regularizer? y / n")
            response = input()
            if response == 'n':
                exit()
            print('Resuming...')
        if self.config_args.get('gradient_clipping'):
            self.gradient_clipping = self.config_args['gradient_clipping']['value']
            logger.info(f"Clipping gradients at value {self.gradient_clipping}")
        else:
            logger.info("Are you sure you don't want to use a gradient clipping? y / n")
            response = input()
            if response == 'n':
                logger.info("which value?")
                response = input()
                self.gradient_clipping = float(response)
            print('Resuming...')

        self.instantiate()

        # Ignite Setup
        self.custom_metrics = {
            "accuracy": Accuracy(),
            "loss": Loss(self.loss.loss),
        }
        self.trainer = self.create_supervised_trainer()
        self.evaluator = self.create_supervised_evaluator(metrics=self.custom_metrics)

        self.dataset.set_split(split='train')
        self.train_loader = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'],
                                                     device=self.config_args['device'])

        self.dataset.set_split(split='val')
        self.val_loader = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'],
                                                   device=self.config_args['device'])

        self.dataset.set_split(split='test')
        self.test_loader = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'],
                                                    device=self.config_args['device'])

        pbar = ProgressBar()
        pbar.attach(self.trainer)
        pbar.attach(self.evaluator)

        # A few events handler. To add / modify the events handler, you need to extend the __init__ method of RunnerABC
        # Ignite provides the necessary abstractions and a furnished repository of useful tools
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            logger.info(f"Training Results - Epoch: {trainer.state.epoch} Acc: {metrics['accuracy']}| Loss: {metrics['loss']}")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            logger.info(f"Validation Results - Epoch: {self.trainer.state.epoch} Acc: {metrics['accuracy']} | Loss: {metrics['loss']}")

            self.scheduler.scheduler.step(metrics['loss'])

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(trainer):
            self.evaluator.run(self.test_loader)
            metrics = self.evaluator.state.metrics
            logger.info(f"Test Results - Epoch: {self.trainer.state.epoch} Acc: {metrics['accuracy']} | Loss: {metrics['loss']}")

    # Methods to load the experiment file and convert it into a Runner object that can be used for training
    @classmethod
    def load_from_project(cls, experiment_file: str, **kwargs: str):
        """
        Instantiate an experiment
        :param experiment_file:
        :param kwargs: substitution args. it's recommended to use all caps to avoid naming conflicts
        :return:
        """

        experiments_path = Path(__file__).resolve().parent.parent
        experiments_path /= experiment_file

        with open(experiments_path, 'r') as exp:
            experiment = json.load(exp)

        for k in list(experiment.keys()):
            v = experiment[k]

            if isinstance(v, str):
                for param_name, param_val in kwargs.items():
                    experiment[k] = v.replace(param_name, param_val)

        return cls(config_args=experiment)

    def instantiate(self):

        # Manage file directories
        if self.config_args['expand_filepaths_to_save_dir']:
            self.config_args['vectorizer_file'] = self.config_args['save_dir'] + '/' + self.config_args['vectorizer_file']

            self.config_args['model_state_file'] = self.config_args['save_dir'] + '/' + self.config_args['model_state_file']

            logger.info("Expanded filepaths: ")
            logger.info(f"{self.config_args['vectorizer_file']}")
            logger.info(f"{self.config_args['model_state_file']}")

        # Initialize training state and cuda device
        self.training_state = make_training_state(args=self.config_args)
        if not torch.cuda.is_available():
            self.config_args['cuda'] = False
        self.config_args['device'] = torch.device("cuda" if self.config_args['cuda'] else "cpu")

        # Set seed for reproducibility
        set_seed_everywhere(self.config_args['seed'], self.config_args['cuda'])

        # handle dirs
        handle_dirs(dirpath=self.config_args['save_dir'])

        # Load dataset and vectorizer
        logger.info("Loading the data and getting the vectorizer ready")
        if self.config_args['reload_from_files']:
            # training from a checkpoint
            self.dataset = self.dataset_cls.load_dataset_and_load_vectorizer(self.config_args['dataset'],
                                                                             self.config_args['vectorizer_file'])
        else:
            # # create dataset and vectorizer
            # if self.config_args.get('load_from_line_file', None):
            #     self.dataset = self.dataset_cls.load_dataset_and_make_vectorizer_from_file(data_file=self.config_args['data_file'])
            # else:
            self.dataset = self.dataset_cls.load_dataset_and_make_vectorizer(self.config_args['dataset'])
            self.dataset.save_vectorizer(self.config_args['vectorizer_file'])
        self.vectorizer = self.dataset.get_vectorizer()

        # Word Embeddings
        if self.config_args.get('use_glove', False):
            words = self.vectorizer.data_vocab._token2id.keys()
            embeddings = make_embedding_matrix(glove_filepath=self.config_args['glove_filepath'],
                                               words=words)
            logging.info("Using pre-trained embeddings")
        else:
            logger.info("Not using pre-trained embeddings")
            embeddings = None

        # Register useful parameters and objects useful for model instantiation #TODO: do proper testing on this part
        self.config_args['pretrained_embeddings'] = embeddings
        self.config_args['num_features'] = self.config_args['input_dim'] = self.config_args['vocabulary_size'] = self.config_args['initial_num_channels'] = \
            self.config_args['num_embeddings'] = self.config_args['char_vocab_size'] = self.config_args['source_vocab_size'] = len(self.vectorizer.data_vocab)
        self.config_args['output_dim'] = self.config_args['num_classes'] = self.config_args['num_nationalities'] = self.config_args['target_vocab_size'] = len(
            self.vectorizer.target_vocab)
        if hasattr(self.vectorizer.data_vocab, 'mask_index'):
            self.config_args['padding_idx'] = self.config_args.get('padding_idx', 0)
        else:
            self.config_args['padding_idx'] = 0  # TODO: see if this fails
        if hasattr(self.vectorizer.target_vocab, 'begin_seq_index'):
            self.config_args['target_bos_index'] = self.vectorizer.target_vocab.begin_seq_index
        self.mask_index = self.vectorizer.data_vocab.mask_index if hasattr(self.vectorizer.data_vocab, 'mask_index') else None
        if hasattr(self.dataset, 'class_weights'):
            self.dataset.class_weights = self.dataset.class_weights.to(self.config_args['device'])
            self.config_args['weight'] = self.dataset.class_weights

        # Model
        self.model: nn.Module = Model.from_config(config_args=self.config_args)
        # Loss
        self.loss: LossFunction = LossFunction(config_args=self.config_args)
        # Optimizer
        self.config_args['params'] = [p for p in self.model.parameters() if p.requires_grad]  # parameters that will be optimized by the optimizer
        self.optimizer: Optimizer = Optimizer(config_args=self.config_args).optimizer
        self.config_args['optimizer'] = self.optimizer
        # Scheduler
        self.scheduler: Scheduler = Scheduler(config_args=self.config_args)
        # Batch generator
        self.generator: Generator = Generator(config_args=self.config_args)
        # Metrics
        self.metrics: Metrics = Metrics(config_args=self.config_args)
        # Regularizer
        if self.config_args.get('Regularizer'):
            self.regularizer: Regularizer = Regularizer(config_args=self.config_args)
            logger.info(f"Using regularizer {self.regularizer}")

    def _prepare_batch(batch: Dict, device=None, non_blocking: bool = False):
        """Prepare batch for training: pass to a device with options.

        """
        result = {key: convert_tensor(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
        return result

    def create_supervised_trainer(self, prepare_batch=_prepare_batch, non_blocking=False):

        if self.config_args['device']:
            self.model.to(self.config_args['device'])

        # Gradient accumulation trick adapted from :
        # https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        accumulation_steps = 4

        def _update(engine, batch):

            self.model.train()

            if engine.state.iteration % accumulation_steps == 0:
                self.optimizer.zero_grad()

            batch = prepare_batch(batch, device=self.config_args['device'], non_blocking=non_blocking)
            model_inputs = {inp: batch[inp] for inp in self.model.inputs_names}
            y_pred = self.model(**model_inputs)

            loss_params = {
                "input": y_pred,
                "target": batch['y_target']}

            if hasattr(self.loss.loss, 'mask') and self.mask_index:
                loss_params['mask_index'] = self.mask_index

            loss = self.loss.loss(**loss_params) / accumulation_steps
            loss.backward()

            if engine.state.iteration % accumulation_steps == 0:
                self.optimizer.step()

            return loss.item()

        return Engine(_update)

    def create_supervised_evaluator(self, metrics: Dict, prepare_batch=_prepare_batch, non_blocking=False):

        if self.config_args['device']:
            self.model.to(self.config_args['device'])

        def _inference(engine, batch):
            self.model.eval()
            with torch.no_grad():
                batch = prepare_batch(batch, device=self.config_args['device'], non_blocking=non_blocking)
                model_inputs = {inp: batch[inp] for inp in self.model.inputs_names}
                y_pred = self.model(**model_inputs)
                return y_pred, batch['y_target']

        engine = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(engine, name)

        return engine

    def run_pipeline(self):
        self.trainer.run(self.train_loader, max_epochs=self.config_args['num_epochs'])

    # Methods used if you don't want to use ignite
    def to_tensorboard(self, epoch: int, metrics: List[str]):

        self.writer.add_scalar('Train/loss', self.training_state['train_loss'][-1], epoch)
        self.writer.add_scalar('Val/loss', self.training_state['val_loss'][-1], epoch)

        for metric in metrics:

            if f"train_{metric}" in self.training_state and f"val_{metric}" in self.training_state:
                self.writer.add_scalar(f"Train/{metric}", self.training_state.get(f"train_{metric}")[-1], epoch)
                self.writer.add_scalar(f"Val/{metric}", self.training_state.get(f"val_{metric}")[-1], epoch)

            else:
                raise NotImplementedError(f"Error {metric} is not implemented yet")

        if hasattr(self.model, "embedding"):
            logger.info("Logging embeddings to Tensorboard!")
            embeddings = self.model.embedding.weight.data
            metadata = [str(self.vectorizer.data_vocab._id2token[token_index]).encode('utf-8') for token_index in range(embeddings.shape[0])]
            self.writer.add_embedding(mat=embeddings, metadata=metadata, global_step=epoch)

    def log_current_metric(self, epoch: int, metrics: List[str]):

        current_metrics = {
            'tl': self.training_state['train_loss'][-1],
            'vl': self.training_state['val_loss'][-1]}

        for metric in metrics:

            if f"train_{metric}" in self.training_state and f"val_{metric}" in self.training_state:
                current_metrics[f"train_{metric}"] = self.training_state[f"train_{metric}"][-1]
                current_metrics[f"val_{metric}"] = self.training_state[f"val_{metric}"][-1]
            else:
                raise NotImplementedError(f"Error {metric} is not implemented yet")
        current_metrics = {key: np.round(value, 3) for key, value in current_metrics.items()}

        logger.info(f"Epoch {epoch}: train loss: {current_metrics['tl']} / val loss: {current_metrics['vl']}")
        for metric in metrics:
            train = current_metrics[f'train_{metric}']
            val = current_metrics[f'val_{metric}']
            logger.info(f"Metric {metric} --> Train: {train} / Val: {val}")

    def log_test_metric(self, metrics: List[str]):

        current_metrics = {
            'tl': self.training_state['test_loss']}

        for metric in metrics:
            if f"test_{metric}" in self.training_state:
                current_metrics[f"test_{metric}"] = self.training_state[f"test_{metric}"][-1]
            else:
                raise NotImplementedError(f"Error {metric} is not implemented yet")
        current_metrics = {key: np.round(value, 3) for key, value in current_metrics.items()}

        logger.info(f"Test loss: {current_metrics['tl']}")
        for metric in metrics:
            test = current_metrics[f'test_{metric}']
            logger.info(f"Test on metric {metric}: {test}")

    def update(self, batch_dict: Dict, running_loss: float, batch_index: int, running_metrics: Dict, compute_gradient: bool = True):
        raise NotImplementedError

    def train_and_validate_one_epoch(self):

        self.epoch_index += 1
        # sample_probability = (20 + self.epoch_index) / self.config_args['num_epochs']  # TODO: include this into the NMT training part

        self.training_state['epoch_index'] += 1

        # Set the dataset object to train mode such that the dataset used is the training data
        self.dataset.set_split(split='train')
        batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])
        running_loss = 0
        running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}
        # Set the model object to train mode (torch optimizes the parameters)
        self.model.train()
        num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])
        for batch_index, batch_dict in tqdm(enumerate(batch_generator), total=num_batch, desc='Training batches'):
            running_loss, running_metrics = self.update(batch_dict=batch_dict, running_loss=running_loss, running_metrics=running_metrics,
                                                        batch_index=batch_index, compute_gradient=True)

        self.training_state['train_loss'].append(running_loss)
        for metric in self.metrics.names:
            self.training_state[f"train_{metric}"].append(running_metrics[f"running_{metric}"])

        # Iterate over validation dataset
        self.dataset.set_split(split='val')
        batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])
        running_loss = 0
        running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}
        # Set the model object to val mode (torch does not optimize the parameters)
        self.model.eval()
        num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])
        for batch_index, batch_dict in tqdm(enumerate(batch_generator), total=num_batch, desc='Validation batches'):
            running_loss, running_metrics = self.update(batch_dict=batch_dict, running_loss=running_loss, running_metrics=running_metrics,
                                                        batch_index=batch_index, compute_gradient=False)

        self.training_state['val_loss'].append(running_loss)
        for metric in self.metrics.names:
            self.training_state[f"val_{metric}"].append(running_metrics[f"running_{metric}"])

        self.training_state = update_train_state(config_args=self.config_args, model=self.model,
                                                 train_state=self.training_state)
        self.scheduler.scheduler.step(self.training_state['val_loss'][-1])

    def run(self, test_at_the_end: bool = False):
        """
        Training loop
        :return:
        """

        # Train/Val loop
        logger.info("Entering the training loop...")

        try:
            for epoch in range(self.config_args['num_epochs']):

                logger.info(f"Epoch {epoch + 1}/{self.config_args['num_epochs']}")
                # self.train_one_epoch()

                self.train_and_validate_one_epoch()

                self.to_tensorboard(epoch=epoch, metrics=self.metrics.names)
                self.log_current_metric(epoch=epoch, metrics=self.metrics.names)
                if self.training_state['stop_early']:
                    break

        except KeyboardInterrupt:
            logger.info("Leaving training phase early (Action taken by user)")

        # Optional testing phase [Not a good practice during development time, use this only when you are sure of your modelling decisions!]
        if test_at_the_end:

            logger.info("Entering the test phase...")

            self.dataset.set_split(split='test')
            batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])
            num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])
            running_loss = 0
            running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}
            self.model.eval()
            for batch_index, batch_dict in tqdm(enumerate(batch_generator), total=num_batch, desc='Test batches'):
                running_loss, running_metrics = self.update(batch_dict=batch_dict, running_loss=running_loss, running_metrics=running_metrics,
                                                            batch_index=batch_index, compute_gradient=False)
            self.training_state['test_loss'] = running_loss
            for metric in self.metrics.names:
                self.training_state[f"test_{metric}"].append(running_metrics[f"running_{metric}"])
            # self.do_test()
            self.log_test_metric(metrics=self.metrics.names)

    # Methods used for transfer learning
    # The pattern for updating the optimizer is adapted from this discussion:
    #  https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/8
    def freeze_params(self, to_freeze: List[str]):
        """
        Freeze all parameters from a sublist of named_parameters
        E.g. self.freeze_params(to_freeze=["fc1.weight"])
        :param to_freeze: list of model parameters to be frozen
        :return:
        """
        # Setting given parameters to requires_grad=False
        logger.info(f"Freezing params {[name for name, parameter in self.model.named_parameters() if name in to_freeze]}")
        [parameter.requires_grad_(False) for name, parameter in self.model.named_parameters() if name in to_freeze]

        # Initializing the optimimzer with the trainable parameters only
        self.config_args['params'] = [p for p in self.model.parameters() if p.requires_grad]  # parameters that will be optimized by the optimizer
        self.optimizer: Optimizer = Optimizer(config_args=self.config_args).optimizer
        self.config_args['optimizer'] = self.optimizer

    def freeze_attr_params(self, to_freeze: List[Any]):
        """
        Freeze all parameters from a sublist of model attributes
        E.g. self.freeze_attr_params(to_freeze=["fc1"])
        :param to_freeze: list of model attributes whose parameters should be frozen
        :return:
        """
        # Setting given parameters to requires_grad=False
        logger.info(f"Freezing attributes {[{attr: [name for name, parameter in self.model.__getattr__(name=attr).named_parameters()]} for attr in to_freeze]}")
        [parameter.requires_grad_(False) for attr in to_freeze for name, parameter in self.model.__getattr__(name=attr).named_parameters()]
        # Initializing the optimimzer with the trainable parameters only
        self.config_args['params'] = [p for p in self.model.parameters() if p.requires_grad]  # parameters that will be optimized by the optimizer
        self.optimizer: Optimizer = Optimizer(config_args=self.config_args).optimizer
        self.config_args['optimizer'] = self.optimizer

    def unfreeze_params(self, to_unfreeze: List[str]):
        """
        Unfreeze all parameters from a sublist of named_parameters
        E.g. self.unfreeze_params(to_freeze=["fc1.weight"])
        :param to_unfreeze: list of model parameters to be unfrozen
        :return:
        """
        # Set the parameters to requires_grad=True
        logger.info(f"Unfreezing params {[name for name, parameter in self.model.named_parameters() if name in to_unfreeze]}")
        [parameter.requires_grad_(True) for name, parameter in self.model.named_parameters() if name in to_unfreeze]
        # Add those parameters to the optimizer's list of parameters to optimize
        for name, parameter in self.model.named_parameters():
            if name in to_unfreeze:
                try:
                    self.optimizer.add_param_group({
                                                       'params': parameter})
                except ValueError as e:
                    logger.info(f"Parameters {name} are already in the optimimzer's list!")
                    logger.info(e)

    def unfreeze_attr_params(self, to_unfreeze: List[Any]):
        """
        Unfreeze all parameters from a sublist of model attributes
        E.g. self.unfreeze_attr_params(to_unfreeze=["fc1"])
        :param to_unfreeze: list of model attributes whose parameters should be unfrozen
        :return:
        """
        # Set the parameters to requires_grad=True
        logger.info(
            f"Unfreezing attributes {[{attr: [name for name, parameter in self.model.__getattr__(name=attr).named_parameters()]} for attr in to_unfreeze]}")
        # logger.info(f"Unfreezing attributes {[name for attr in to_unfreeze for name, parameter in self.model.__getattr__(name=attr).named_parameters()]}")
        [parameter.requires_grad_(True) for attr in to_unfreeze for name, parameter in self.model.__getattr__(name=attr).named_parameters()]
        # Add those parameters to the optimizer's list of parameters to optimize
        for attr in to_unfreeze:
            for name, parameter in self.model.__getattr__(name=attr).named_parameters():
                try:
                    self.optimizer.add_param_group({
                                                       'params': parameter})
                except ValueError as e:
                    logger.info(f"Parameters {name} from attribute {attr} are already in the optimimzer's list!")
                    logger.info(e)


def build_experiment(config: str):
    experiments_path = Path(__file__).resolve().parent.parent / 'experiments'
    experiments_path /= config
    print(experiments_path)

    with open(experiments_path, 'r') as exp:
        experiment = json.load(exp)

    runner = RunnerABC(config_args=experiment)
    return runner


if __name__ == "__main__":
    experiment = "experiments/mlp.json"
    runner = RunnerABC.load_from_project(experiment_file=experiment)
    runner.run_pipeline()
