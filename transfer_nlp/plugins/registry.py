"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""

from typing import Dict, List
import logging

import torch.nn as nn
import torch.optim as optim
import inspect

name = 'transfer_nlp.plugins.registry'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)

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
ACTIVATION_FUNCTIONS = {
    "ReLU": nn.functional.relu,
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
    if activation_class.__name__ in ACTIVATION_FUNCTIONS:
        existing = list(ACTIVATION_FUNCTIONS.keys())
        raise ValueError(f"{activation_class.__name__} is already registered. Please have a look at the existing activation functions: {existing}")
    else:
        ACTIVATION_FUNCTIONS[activation_class.__name__] = activation_class
        return activation_class


def register_model(model_class):
    if model_class.__name__ in MODEL_CLASSES:
        existing = list(MODEL_CLASSES.keys())
        raise ValueError(f"{model_class.__name__} is already registered. Please have a look at the existing models: {existing}")
    else:
        MODEL_CLASSES[model_class.__name__] = model_class
        return model_class


def register_loss(loss_class):
    if loss_class.__name__ in LOSS_CLASSES:
        existing = list(LOSS_CLASSES.keys())
        raise ValueError(f"{loss_class.__name__} is already registered. Please have a look at the existing models: {existing}")
    else:
        LOSS_CLASSES[loss_class.__name__] = loss_class
        return loss_class


def register_optimizer(optimizer_class):
    if optimizer_class.__name__ in OPTIMIZER_CLASSES:
        existing = list(OPTIMIZER_CLASSES.keys())
        raise ValueError(f"{optimizer_class.__name__} is already registered. Please have a look at the existing optimizers: {existing}")
    else:
        OPTIMIZER_CLASSES[optimizer_class.__name__] = optimizer_class
        return optimizer_class


def register_scheduler(scheduler_class):
    if scheduler_class.__name__ in SCHEDULER_CLASSES:
        existing = list(SCHEDULER_CLASSES.keys())
        raise ValueError(f"{scheduler_class.__name__} is already registered. Please have a look at the existing schedulers: {existing}")
    else:
        SCHEDULER_CLASSES[scheduler_class.__name__] = scheduler_class
        return scheduler_class


def register_dataset(dataset_class):
    if dataset_class.__name__ in DATASET_CLASSES:
        existing = list(DATASET_CLASSES.keys())
        raise ValueError(f"{dataset_class.__name__} is already registered. Please have a look at the existing dataset classes: {existing}")
    else:
        DATASET_CLASSES[dataset_class.__name__] = dataset_class
        return dataset_class


def register_batch_generator(batch_generator_class):
    if batch_generator_class.__name__ in BATCH_GENERATORS:
        existing = list(BATCH_GENERATORS.keys())
        raise ValueError(f"{batch_generator_class.__name__} is already registered. Please have a look at the existing batch generators: {existing}")
    else:
        BATCH_GENERATORS[batch_generator_class.__name__] = batch_generator_class
        return batch_generator_class


def register_metric(metric_class):
    if metric_class.__name__ in METRICS:
        existing = list(METRICS.keys())
        raise ValueError(f"{metric_class.__name__} is already registered. Please have a look at the existing metrics: {existing}")
    else:
        METRICS[metric_class.__name__] = metric_class
        return metric_class


def register_regularizer(regularizer_class):
    if regularizer_class.__name__ in REGULARIZER_CLASSES:
        existing = list(REGULARIZER_CLASSES.keys())
        raise ValueError(f"{regularizer_class.__name__} is already registered. Please have a look at the existing metrics: {existing}")
    else:
        REGULARIZER_CLASSES[regularizer_class.__name__] = regularizer_class
        return regularizer_class


class Metrics:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.names = config_args['metrics']
        self.metrics = {}
        existing = list(METRICS.keys())
        for name in self.names:
            try:
                self.metrics[name] = METRICS[name]
            except KeyError as k:
                raise KeyError(
                    f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                    f"Please check your implementation and experiment config file match")


class Generator:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        name = config_args['batch_generator']
        existing = list(BATCH_GENERATORS.keys())
        try:
            self.generator = BATCH_GENERATORS[name]
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")


class Data:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        name = config_args['dataset_cls']
        existing = list(DATASET_CLASSES.keys())
        try:
            self.dataset = DATASET_CLASSES[name]
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")


class Model(nn.Module):
    """
    The aim of this class is to load a PyTorch nn.Module model from a json file containing a model definition as:

    "model": {
    "modelName": "MultiLayerPerceptron",
    "modelParams": [
      "input_dim",
      "hidden_dim",
      "output_dim"
    ],
    "modelInputs": [
      "x_in"
    ]
  }
   The motivation for this is that we want the framework to be usable with any model with custom signature.
   Please note that the input variables names should match the outputs of te batch generators, as the Model will
   directly take tne generator's output as input to the forward method.

    """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    @classmethod
    def from_config(cls, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        config_args: Dict = config_args
        try:
            name = config_args['model']['modelName']
            params_names = config_args['model']['modelParams']
            inputs_names = config_args['model']['modelInputs']
        except KeyError as k:
            raise KeyError(f"{k} is missing from the config file")

        existing = list(MODEL_CLASSES.keys())

        model_params = {}
        for param in params_names:
            try:
                # Some parameters of the model cannot be set in advance and have to be computed in the runner instantiation step
                # For example if a model needs a vocabulary size, this might be computed by the data loader, and then this value
                # will be accessible in the config dictionary, which we pull here:
                model_params[param] = config_args[param]
            except KeyError as k:
                raise KeyError(f"{k} is not a parameter for {name}, Please check your implementation and experiment config file match")

        try:
            model = MODEL_CLASSES[name](**model_params)
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {cls.__name__}: {existing}. "
                f"Please check that your implementation and experiment config file match")

        # Get the names of variables for model inputs and store them
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(model.forward)

        if defaults:
            optional_args_start = len(args) - len(defaults)
            options = {args[optional_args_start + i]: defaults[i] for i in range(len(defaults))}

            args = args[:-len(defaults)]
            logger.info(f"Inputs of forward method: {args[1:]}")
            logger.info(f"Optional inputs parameters of forward method: {options}")
        for arg in args[1:]:  # index 0 is 'self' so we don't use it
            if arg not in inputs_names:
                raise ValueError(f"{arg} does not match any of the model input names")

        # We decorate the nn.Module model with names of the model class, param names and input names
        model.name = name
        model.params_names = params_names
        model.inputs_names = inputs_names
        logger.info("Using the following classifier:")
        logger.info(f"{model}")
        model = model.to(config_args['device'])
        return model

    def forward(self, *input, **kwargs):
        return self.forward(*input, **kwargs)


class LossFunction:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        try:
            self.name: str = self.config_args['loss']['lossName']
            self.params: List[str] = self.config_args['loss']['lossParams']
        except KeyError as k:
            raise KeyError(f"{k} is missing from the config file")

        existing = list(LOSS_CLASSES.keys())

        loss_params = {}
        for param in self.params:
            try:
                # Same comment as for model definition
                loss_params[param] = self.config_args[param]
            except KeyError as k:
                raise KeyError(f"{k} is not a parameter for {self.name}, Please check your implementation and experiment config file match")

        try:
            self.loss = LOSS_CLASSES[self.name](**loss_params)
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")


class Optimizer:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        try:
            self.name: str = self.config_args['Optimizer']['optimizerName']
            self.params: List[str] = self.config_args['Optimizer']['optimizerParams']
        except KeyError as k:
            raise KeyError(f"{k} is missing from the config file")

        existing = list(OPTIMIZER_CLASSES.keys())

        optimizer_params = {}
        for param in self.params:
            try:
                # Same comment as for model definition
                optimizer_params[param] = self.config_args[param]
            except KeyError as k:
                raise KeyError(f"{k} is not a parameter for {self.name}, Please check your implementation and experiment config file match")

        try:
            self.optimizer = OPTIMIZER_CLASSES[self.name](**optimizer_params)
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")


class Scheduler:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        try:
            self.name: str = self.config_args['scheduler']['schedulerName']
            self.params: List[str] = self.config_args['scheduler']['schedulerParams']
        except KeyError as k:
            raise KeyError(f"{k} is missing from the config file")

        existing = list(SCHEDULER_CLASSES.keys())

        scheduler_params = {}
        for param in self.params:
            try:
                # Same comment as for model definition
                scheduler_params[param] = self.config_args[param]
            except KeyError as k:
                raise KeyError(f"{k} is not a parameter for {self.name}, Please check your implementation and experiment config file match")

        try:
            self.scheduler = SCHEDULER_CLASSES[self.name](**scheduler_params)
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")


class Regularizer:

    def __init__(self, config_args: Dict):
        """
        :param config_args: Contains the experiment configuration, with all necessary hyperparameters
        """
        self.config_args: Dict = config_args
        try:
            self.name: str = self.config_args['Regularizer']['regularizerName']
            self.params: List[str] = self.config_args['Regularizer']['regularizerParams']
        except KeyError as k:
            raise KeyError(f"{k} is missing from the config file")

        existing = list(REGULARIZER_CLASSES.keys())

        regularizer_params = {}
        for param in self.params:
            try:
                # Same comment as for model definition
                regularizer_params[param] = self.config_args[param]
            except KeyError as k:
                raise KeyError(f"{k} is not a parameter for {self.name}, Please check your implementation and experiment config file match")

        try:
            self.regularizer = REGULARIZER_CLASSES[self.name](**regularizer_params)
        except KeyError as k:
            raise KeyError(
                f"{k} is not among the registered {self.__class__.__name__}: {existing}. "
                f"Please check your implementation and experiment config file match")

    def __str__(self):
        return self.regularizer.__str__()
