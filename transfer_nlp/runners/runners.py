import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np

from embeddings.embeddings import make_embedding_matrix
from loaders.loaders import ReviewsDataset, SurnamesDataset, SurnamesDatasetCNN, CBOWDataset, NewsDataset, \
    CustomDataset, SurnameDatasetRNN, \
    SurnameDatasetGeneration, NMTDataset, FeedlyDataset
from loaders.vectorizers import Vectorizer
from models.cbow import CBOWClassifier
from models.cnn import SurnameClassifierCNN, NewsClassifier
from models.generation import SurnameConditionedGenerationModel
from models.nmt import NMTModel
from models.perceptrons import MultiLayerPerceptron, Perceptron
from models.rnn import SurnameClassifierRNN
from runners.utils import set_seed_everywhere, handle_dirs, make_training_state

name = 'transfer_nlp.runners.runners'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')

DATASET_CLASSES = {'NewsDataset': NewsDataset,
                   'CBOWDataset': CBOWDataset,
                   'SurnamesDatasetCNN': SurnamesDatasetCNN,
                   'SurnamesDataset': SurnamesDataset,
                   'ReviewsDataset': ReviewsDataset,
                   'SurnameDatasetRNN': SurnameDatasetRNN,
                   'SurnameDatasetGeneration': SurnameDatasetGeneration,
                   'NMTDataset': NMTDataset,
                   'FeedlyDataset': FeedlyDataset}

MODEL_CLASSES = {'NewsClassifier': NewsClassifier,
                 'CBOWClassifier': CBOWClassifier,
                 'SurnameClassifierCNN': SurnameClassifierCNN,
                 'MultiLayerPerceptron': MultiLayerPerceptron,
                 'Perceptron': Perceptron,
                 'SurnameClassifierRNN': SurnameClassifierRNN,
                 'SurnameConditionedGenerationModel': SurnameConditionedGenerationModel,
                 'NMTModel': NMTModel}

class RunnerParams:

    def __init__(self):
        pass

class ModelParams:

    def __init__(self):
        pass


class Model:

    def __init__(self, params: ModelParams):
        pass
    @classmethod
    def from_params(cls, params: ModelParams):
        return cls(params=params)

class RunnerABC:

    def __init__(self, params: RunnerParams):

        self.dataset_cls = DATASET_CLASSES[params.dataset_cls]
        self.model: nn.Module = MODEL_CLASSES[params.model]

        self.args: RunnerParams = params
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
        self.writer = SummaryWriter(log_dir=params.logs)

        self.instantiate()
        pass

    @classmethod
    def from_params(cls, params: RunnerParams) -> 'RunnerABC':
        return cls(params=params)

    def instantiate_model(self):
        self.model = Model.from

    def instantiate(self):

        if self.args.expand_filepaths_to_save_dir:
            self.args.vectorizer_file = self.args.save_dir + '/' + self.args.vectorizer_file

            self.args.model_state_file = self.args.save_dir + '/' + self.args.model_state_file

            logger.info("Expanded filepaths: ")
            logger.info(f"{self.args.vectorizer_file}")
            logger.info(f"{self.args.model_state_file}")

        self.training_state = make_training_state(args=self.args)

        if not torch.cuda.is_available():
            self.args.cuda = False
        self.args.device = torch.device("cuda" if self.args.cuda else "cpu")

        # Set seed for reproducibility
        set_seed_everywhere(self.args.seed, self.args.cuda)

        # handle dirs
        handle_dirs(self.args.save_dir)

        # Load dataset and vectorizer
        logger.info("Loading the data and getting the vectorizer ready")

        if self.args.reload_from_files:
            # training from a checkpoint
            logger.info("Loading dataset and vectorizer")
            self.dataset = self.dataset_cls.load_dataset_and_load_vectorizer(self.args.data_csv,
                                                                        self.args.vectorizer_file)
        else:
            logger.info("Loading dataset and creating vectorizer")
            # create dataset and vectorizer
            self.dataset = self.dataset_cls.load_dataset_and_make_vectorizer(self.args.data_csv)
            self.dataset.save_vectorizer(self.args.vectorizer_file)

        self.vectorizer = self.dataset.get_vectorizer()

        ##### Instantiate classifier #####

        # Use GloVe or randomly initialized embeddings
        if self.args.use_glove:
            words = self.vectorizer.data_vocab._token2id.keys()
            embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                               words=words)
            logging.info("Using pre-trained embeddings")
        else:
            logger.info("Not using pre-trained embeddings")
            embeddings = None

        if self.model == Perceptron:
            self.model = Perceptron(num_features=len(self.vectorizer.data_vocab))  # 1 1-layer perceptron
        elif self.model == MultiLayerPerceptron:
            self.model = MultiLayerPerceptron(input_dim=len(self.vectorizer.data_vocab), hidden_dim=self.args.hidden_dim,
                                              output_dim=len(self.vectorizer.target_vocab))
            # self.model = MultiLayerPerceptron(input_dim=len(self.vectorizer.data_vocab), hidden_dim=self.args.hidden_dim, output_dim=1)  #2 MLP
        elif self.model == SurnameClassifierCNN:
            self.model = SurnameClassifierCNN(initial_num_channels=len(self.vectorizer.data_vocab),
                                              num_classes=len(self.vectorizer.target_vocab),
                                              num_channels=self.args.num_channels)
        elif self.model == CBOWClassifier:
            self.model = CBOWClassifier(vocabulary_size=len(self.vectorizer.data_vocab),
                                        embedding_size=self.args.embedding_size)
        elif self.model == NewsClassifier:
            self.model = NewsClassifier(embedding_size=self.args.embedding_size,
                                             num_embeddings=len(self.vectorizer.data_vocab),
                                             num_channels=self.args.num_channels,
                                             hidden_dim=self.args.hidden_dim,
                                             num_classes=len(self.vectorizer.target_vocab),
                                             dropout_p=self.args.dropout_p,
                                             pretrained_embeddings=embeddings,
                                             padding_idx=0)
        elif self.model == SurnameClassifierRNN:
            self.model = SurnameClassifierRNN(embedding_size=self.args.char_embedding_size,
                               num_embeddings=len(self.vectorizer.data_vocab),
                               num_classes=len(self.vectorizer.target_vocab),
                               rnn_hidden_size=self.args.rnn_hidden_size,
                               padding_idx=self.vectorizer.data_vocab.mask_index)

        elif self.model == SurnameConditionedGenerationModel:
            self.model = SurnameConditionedGenerationModel(char_embedding_size=self.args.char_embedding_size,
                                   char_vocab_size=len(self.vectorizer.data_vocab),
                                   num_nationalities=len(self.vectorizer.target_vocab),
                                   rnn_hidden_size=self.args.rnn_hidden_size,
                                   padding_idx=self.vectorizer.data_vocab.mask_index,
                                   dropout_p=0.5,
                                   conditioned=self.args.conditioned)
            self.mask_index = self.vectorizer.data_vocab.mask_index
            self.is_pred_continuous = False

        elif self.model == NMTModel:

            self.model = NMTModel(source_vocab_size=len(self.vectorizer.data_vocab),
                             source_embedding_size=self.args.source_embedding_size,
                             target_vocab_size=len(self.vectorizer.target_vocab),
                             target_embedding_size=self.args.target_embedding_size,
                             encoding_size=self.args.encoding_size,
                             target_bos_index=self.vectorizer.target_vocab.begin_seq_index)
            self.mask_index = self.vectorizer.data_vocab.mask_index
            self.is_pred_continuous = False

        else:
            logger.info("You must first design a model and then use it as argument")


        logger.info("Using the following classifier:")
        logger.info(f"{self.model}")
        self.model = self.model.to(self.args.device)

        # Define loss function and optimizer
        if self.dataset_cls == ReviewsDataset:
            self.loss_func = nn.BCEWithLogitsLoss()

        elif self.dataset_cls in [SurnamesDataset, SurnamesDatasetCNN, NewsDataset, SurnameDatasetRNN]:
            self.dataset.class_weights = self.dataset.class_weights.to(self.args.device)
            self.loss_func = nn.CrossEntropyLoss(weight=self.dataset.class_weights)
            self.is_output_continuous = False  # This is for the runner to take into account different loss functions, whether they ouput logits or not

        elif self.dataset_cls == CBOWDataset:
            self.loss_func = nn.CrossEntropyLoss()
            self.is_output_continuous = False
            self.is_pred_continuous = False

        elif self.dataset_cls in [SurnameDatasetGeneration, NMTDataset, FeedlyDataset]:
            pass

        else:
            raise ValueError("Only Yelp Reviews, Surnames and CBOW are available at the moment")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)

        if self.dataset_cls == NewsDataset:
            self.is_output_continuous = False
            self.is_pred_continuous = False

    def train_one_epoch(self):
        raise NotImplementedError

    def do_test(self):
        raise NotImplementedError

    def to_tensorboard(self, epoch: int, metric: str='acc'):

        if metric == 'acc':

            self.writer.add_scalar('Train/acc', self.training_state['train_acc'][-1], epoch)
            self.writer.add_scalar('Train/loss', self.training_state['train_loss'][-1], epoch)
            self.writer.add_scalar('Val/acc', self.training_state['val_acc'][-1], epoch)
            self.writer.add_scalar('Val/loss', self.training_state['val_loss'][-1], epoch)

        else:
            raise NotImplementedError("Error metric others than accuracy are not implemented yest")

    def log_current_metric(self, metric: str='acc'):

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

    def run(self, test_at_the_end: bool=False):
        """
        Training loop
        :return:
        """

        # Train/Val loop
        logger.info("#" * 50)
        logger.info("Entering the training loop...")
        logger.info("#" * 50)

        try:
            for epoch in range(self.args.num_epochs):

                logger.info("#" * 50)
                logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
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
