from typing import Dict, List

import torch.nn as nn
import torch.optim as optim

from loaders.loaders import ReviewsDataset, SurnamesDataset, SurnamesDatasetCNN, CBOWDataset, NewsDataset, \
    SurnameDatasetRNN, \
    SurnameDatasetGeneration, NMTDataset, FeedlyDataset
from models.cbow import CBOWClassifier
from models.cnn import SurnameClassifierCNN, NewsClassifier
from models.generation import SurnameConditionedGenerationModel, ConditionedGenerationModel
from models.nmt import NMTModel
from models.perceptrons import MultiLayerPerceptron, Perceptron
from models.rnn import SurnameClassifierRNN
from runners.utils import sequence_loss
from loaders.loaders import generate_nmt_batches, generate_batches
from runners.utils import compute_accuracy_sequence, compute_accuracy


class SequenceLoss:

    def __init__(self):
        self.mask: bool = True

    def __call__(self, *args, **kwargs):
        return sequence_loss(*args, **kwargs)


MODEL_CLASSES = {
    'NewsClassifier': NewsClassifier,
    'CBOWClassifier': CBOWClassifier,
    'SurnameClassifierCNN': SurnameClassifierCNN,
    'MultiLayerPerceptron': MultiLayerPerceptron,
    'Perceptron': Perceptron,
    'SurnameClassifierRNN': SurnameClassifierRNN,
    'SurnameConditionedGenerationModel': SurnameConditionedGenerationModel,
    'ConditionedGenerationModel': ConditionedGenerationModel,
    'NMTModel': NMTModel}

LOSS_CLASSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'sequenceLoss': SequenceLoss,
}

OPTIMIZER_CLASSES = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "AdaDelta": optim.Adadelta,
    "AdaGrad": optim.Adagrad,
    "SparseAdam": optim.SparseAdam,
    "AdaMax": optim.Adamax,
    "ASGD": optim.ASGD,
    "LBFGS": optim.LBFGS,
    "RMSPROP": optim.RMSprop,
    "Rprop": optim.Rprop


}

SCHEDULER_CLASSES = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR

}

DATASET_CLASSES = {
    'NewsDataset': NewsDataset,
    'CBOWDataset': CBOWDataset,
    'SurnamesDatasetCNN': SurnamesDatasetCNN,
    'SurnamesDataset': SurnamesDataset,
    'ReviewsDataset': ReviewsDataset,
    'SurnameDatasetRNN': SurnameDatasetRNN,
    'SurnameDatasetGeneration': SurnameDatasetGeneration,
    'NMTDataset': NMTDataset,
    'FeedlyDataset': FeedlyDataset}

BATCH_GENERATORS = {
    'regular': generate_batches,
    'nmt': generate_nmt_batches,
}

METRICS = {
    "accuracySequence": compute_accuracy_sequence,
    "accuracy": compute_accuracy,
}


class Metric:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        name = config_args['metric']
        self.metric = METRICS[name]


class Generator:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        name = config_args['batch_generator']
        self.generator = BATCH_GENERATORS[name]


class Data:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        name = config_args['dataset_cls']
        self.dataset = DATASET_CLASSES[name]


class Model:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        self.name: str = self.config_args['model']['modelName']
        self.params: List[str] = self.config_args['model']['modelParams']

        model_params = {}
        for param in self.params:
            try:
                model_params[param] = self.config_args.get(param, None)
            except:
                raise ValueError(f"You need to specify the parameter {param} in the config file, of have the runner compute it")

        self.model = MODEL_CLASSES[self.name](**model_params)


class Loss:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        self.name: str = self.config_args['loss']['lossName']
        self.params: List[str] = self.config_args['loss']['lossParams']

        loss_params = {}
        for param in self.params:
            try:
                loss_params[param] = self.config_args[param]
            except:
                raise ValueError(f"You need to specify the parameter {param} in the config file, of have the runner compute it")

        self.loss = LOSS_CLASSES[self.name](**loss_params)


class Optimizer:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        self.name: str = self.config_args['Optimizer']['optimizerName']
        self.params: List[str] = self.config_args['Optimizer']['optimizerParams']

        optimizer_params = {}
        for param in self.params:
            if self.config_args.get(param):
                optimizer_params[param] = self.config_args[param]
            else:
                raise ValueError(f"You need to specify the parameter {param} in the config file, of have the runner compute it")

        self.optimizer = OPTIMIZER_CLASSES[self.name](**optimizer_params)


class Scheduler:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        self.name: str = self.config_args['scheduler']['schedulerName']
        self.params: List[str] = self.config_args['scheduler']['schedulerParams']

        scheduler_params = {}
        for param in self.params:
            if self.config_args.get(param):
                scheduler_params[param] = self.config_args[param]
            else:
                raise ValueError(f"You need to specify the parameter {param} in the config file, of have the runner compute it")

        self.scheduler = SCHEDULER_CLASSES[self.name](**scheduler_params)