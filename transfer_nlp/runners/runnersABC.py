"""
This class contains abstraction interfaces to customize runners.

Ideally, a runner can work with any model. Hence, we instantiate the model parameters names in the experiment json file
{...,
"model": {"modelName": modelName,
          "paramNames": ["input_dim", "output_dim", "embeddings_size"]
          },
...
}

The runner will build a Model object, based on the config file. The config file gives acces to model names and parameters names, but those parameter values
might not be known in advance. Hence, The runner will add its own arguments during model runner instantiation (e.g. vocabulary size, max sequence length, etc...)

"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from embeddings.embeddings import make_embedding_matrix
from loaders.loaders import CustomDataset
from loaders.vectorizers import Vectorizer
from runners.instantiations import Scheduler, Loss, Model, Optimizer, Data
from runners.utils import set_seed_everywhere, handle_dirs, make_training_state

name = 'transfer_nlp.runners.runners'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class RunnerABC:

    def __init__(self, config_args: Dict):

        self.config_args = config_args

        self.dataset_cls = Data(config_args=self.config_args).dataset

        self.dataset: CustomDataset = None
        self.training_state: Dict = {}
        self.vectorizer: Vectorizer = None
        self.loss_func: nn.modules.loss._Loss = None
        self.optimizer: optim.optimizer.Optimizer = None
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = None
        self.is_output_continuous = True
        self.is_pred_continuous = True
        self.mask_index: int = None
        self.epoch_index: int = 0
        self.writer = SummaryWriter(log_dir=self.config_args['logs'])
        self.loss: Loss = None
        self.model: Model = None

        self.instantiate()

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
            logger.info("Loading dataset and vectorizer")
            self.dataset = self.dataset_cls.load_dataset_and_load_vectorizer(self.config_args['data_csv'],
                                                                             self.config_args['vectorizer_file'])
        else:
            logger.info("Loading dataset and creating vectorizer")
            # create dataset and vectorizer
            self.dataset = self.dataset_cls.load_dataset_and_make_vectorizer(self.config_args['data_csv'])
            self.dataset.save_vectorizer(self.config_args['vectorizer_file'])
        self.vectorizer = self.dataset.get_vectorizer()

        # Word Embeddings
        if self.config_args['use_glove']:
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
        self.is_pred_continuous = self.config_args['is_pred_continuous']
        if hasattr(self.dataset, 'class_weights'):
            self.dataset.class_weights = self.dataset.class_weights.to(self.config_args['device'])
            self.config_args['weight'] = self.dataset.class_weights

        # Model
        self.model: Model = Model(config_args=self.config_args)
        logger.info("Using the following classifier:")
        logger.info(f"{self.model.model}")
        self.model = self.model.model.to(self.config_args['device'])

        # Loss, Optimizer and Scheduler
        self.loss: Loss = Loss(config_args=self.config_args)
        self.config_args['params'] = self.model.parameters()
        self.optimizer: Optimizer = Optimizer(config_args=self.config_args)
        self.config_args['optimizer'] = self.optimizer.optimizer
        self.scheduler: Scheduler = Scheduler(config_args=self.config_args)

    def train_one_epoch(self):
        raise NotImplementedError

    def do_test(self):
        raise NotImplementedError

    def to_tensorboard(self, epoch: int, metric: str = 'acc'):

        if metric == 'acc':

            self.writer.add_scalar('Train/acc', self.training_state['train_acc'][-1], epoch)
            self.writer.add_scalar('Train/loss', self.training_state['train_loss'][-1], epoch)
            self.writer.add_scalar('Val/acc', self.training_state['val_acc'][-1], epoch)
            self.writer.add_scalar('Val/loss', self.training_state['val_loss'][-1], epoch)

        else:
            raise NotImplementedError("Error metric others than accuracy are not implemented yest")

    def log_current_metric(self, metric: str = 'acc'):

        if metric == 'acc':
            tp = {
                "tl": self.training_state['train_loss'][-1],
                'ta': self.training_state['train_acc'][-1],
                'vl': self.training_state['val_loss'][-1],
                'va': self.training_state['val_acc'][-1]}
            tp = {key: np.round(value, 3) for key, value in tp.items()}
            logger.info(f"Epoch {epoch}: train loss: {tp['tl']} / val loss: {tp['vl']} / train acc: {tp['ta']} / val acc: {tp['va']}")
        else:
            raise NotImplementedError("Error metric others than accuracy are not implemented yest")

    def log_test_metric(self, metric: str = 'acc'):

        if metric == 'acc':
            tp = {
                "tl": self.training_state['test_loss'][-1],
                'ta': self.training_state['test_acc'][-1]}
            tp = {key: np.round(value, 3) for key, value in tp.items()}
            logger.info(f"Epoch {epoch}: test loss: {tp['tl']} / test acc: {tp['ta']}")
        else:
            raise NotImplementedError("Error metric others than accuracy are not implemented yest")

    def run(self, test_at_the_end: bool = False):
        """
        Training loop
        :return:
        """

        # Train/Val loop
        logger.info("#" * 50)
        logger.info("Entering the training loop...")
        logger.info("#" * 50)

        try:
            for epoch in range(self.config_args['num_epochs']):

                logger.info("#" * 50)
                logger.info(f"Epoch {epoch + 1}/{self.config_args['num_epochs']}")
                logger.info("#" * 50)
                self.train_one_epoch()
                self.to_tensorboard(epoch=epoch, metric='acc')
                self.log_current_metric(metric='acc')
                if self.training_state['stop_early']:
                    break

        except KeyboardInterrupt:
            logger.info("Leaving training phase early (Action taken by user)")

        # Optional testing phase [Not recommended during development time!]
        if test_at_the_end:
            logger.info("#" * 50)
            logger.info("Entering the test phase...")
            logger.info("#" * 50)
            self.do_test()
            logger.info(f"test loss: {self.training_state['test_loss']} / test acc: {self.training_state['test_acc']}")


def build_experiment(config: str):

    experiments_path = Path(__file__).resolve().parent.parent / 'experiments'
    experiments_path /= config
    print(experiments_path)

    with open(experiments_path, 'r') as exp:
        experiment = json.load(exp)

    runner = RunnerABC(config_args=experiment)
    return runner


if __name__ == "__main__":

    experiment = "mlp.json"
    runner = build_experiment(config=experiment)