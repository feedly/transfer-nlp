"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""

import logging
from typing import Dict, List

import torch.nn as nn
import torch.optim as optim

name = 'transfer_nlp.plugins.registry'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')

MODEL_CLASSES = {}
LOSS_CLASSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss, }
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
    "Rprop": optim.Rprop,
}
SCHEDULER_CLASSES = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR

}
DATASET_CLASSES = {}
REGULARIZER_CLASSES = {}
BATCH_GENERATORS = {}
METRICS = {}
ACTIVATION_FUNCTIONS = {"ReLU": nn.functional.relu,
                        "LeakyReLU": nn.functional.leaky_relu,
                        "Tanh": nn.functional.tanh,
                        "Softsign": nn.functional.softsign,
                        "Softshrink": nn.functional.softshrink,
                        "Softplus": nn.functional.softplus,
                        "Sigmoid": nn.Sigmoid,
                        "CELU": nn.CELU,
                        "SELU": nn.functional.selu,
                        "RReLU": nn.functional.rrelu,
                        "ReLU6": nn.functional.relu6,
                        "PReLU": nn.functional.prelu,
                        "LogSigmoid": nn.functional.logsigmoid,
                        "Hardtanh": nn.functional.hardtanh,
                        "Hardshrink": nn.functional.hardshrink,
                        "ELU": nn.functional.elu,
                        "Softmin": nn.functional.softmin,
                        "Softmax": nn.functional.softmax,
                        "LogSoftmax": nn.functional.log_softmax,
                        "GLU": nn.functional.glu,
                        "TanhShrink": nn.functional.tanhshrink
                        }

def register_activation(activation_class):
    # logger.info(f"Registering activation function class {activation_class.__name__} into registry")
    ACTIVATION_FUNCTIONS[activation_class.__name__] = activation_class
    return activation_class


def register_model(model_class):
    # logger.info(f"Registering model {model_class.__name__} into registry")
    MODEL_CLASSES[model_class.__name__] = model_class
    return model_class


def register_loss(loss_class):
    LOSS_CLASSES[loss_class.__name__] = loss_class
    return loss_class


def register_optimizer(optimizer_class):
    OPTIMIZER_CLASSES[optimizer_class.__name__] = optimizer_class
    return optimizer_class


def register_scheduler(scheduler_class):
    MODEL_CLASSES[scheduler_class.__name__] = scheduler_class
    return scheduler_class


def register_dataset(dataset_class):
    DATASET_CLASSES[dataset_class.__name__] = dataset_class
    return dataset_class


def register_batch_generator(batch_generator_class):
    BATCH_GENERATORS[batch_generator_class.__name__] = batch_generator_class
    return batch_generator_class


def register_metric(metric_class):
    METRICS[metric_class.__name__] = metric_class
    return metric_class

def register_regularizer(regularizer_class):
    REGULARIZER_CLASSES[regularizer_class.__name__] = regularizer_class
    return regularizer_class


class Metrics:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.names = config_args['metrics']
        self.metrics = {name: METRICS[name] for name in self.names}


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


class Regularizer:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        self.name: str = self.config_args['Regularizer']['regularizerName']
        self.params: List[str] = self.config_args['Regularizer']['regularizerParams']

        regularizer_params = {}
        for param in self.params:
            if self.config_args.get(param):
                regularizer_params[param] = self.config_args[param]
            else:
                raise ValueError(f"You need to specify the parameter {param} in the config file, of have the runner compute it")

        self.regularizer = REGULARIZER_CLASSES[self.name](**regularizer_params)
