"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union
import logging

import torch.nn as nn
import torch.optim as optim
import inspect
from smart_open import open
name = 'transfer_nlp.plugins.registry'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)

CLASSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
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
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR,
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

def register_plugin(clazz):
    if clazz.__name__ in CLASSES:
        raise ValueError(f"{clazz.__name__} is already registered to class {CLASSES[clazz.__name__]}. Please select another name")
    else:
        CLASSES[clazz.__name__] = clazz
        return clazz

class UnconfiguredItemsException(Exception):
    def __init__(self, items):
        super().__init__('unconfigured items')
        self.items = items

class ExperimentConfig:

    def __init__(self):
        pass

    @staticmethod
    def from_json(experiment:Union[str, Path, Dict]):
        if isinstance(experiment, dict):
            config = experiment
        else:
            config = json.load(open(experiment))

        #extract simple parameters
        experiment = {k:v for k,v in config.items() if not isinstance(v, dict) and not isinstance(v, list)}

        #extract simple lists
        experiment.update({k:v for k,v in config.items() if isinstance(v, list) and all(not isinstance(vv, dict) and not isinstance(vv, list) for vv in v)})

        for k in experiment:
            del config[k]

        while config:
            configured = set()

            unconfigured = {}
            for k,v in config.items():
                if not isinstance(v, dict):
                    raise ValueError(f'complex configuration object config[{k}] must be a dict')

                if '_name' not in v:
                    raise ValueError(f'complex configuration object config[{k}] must be have a "_name" property')

                clazz = CLASSES.get(v['_name'])
                if not clazz:
                    raise ValueError(f'config[{k}] is named {v["_name"]} but this name is not registered. see transfer_nlp.config.register_plugin for more information')

                spec = inspect.getfullargspec(clazz.__init__)
                params = {}

                named_params = {p: pv for p, pv in v.items() if p != '_name'}
                literal_params = {p[:-1]: pv for p, pv in v.items() if p[-1] == '_'}

                for arg in spec.args[1:]:
                    if arg in literal_params:
                        params[arg] = literal_params[arg]
                    else:
                        if arg in named_params:
                            alias = named_params[arg]
                            if alias in experiment:
                                params[arg] = experiment[alias]
                        elif arg in experiment:
                            params[arg] = experiment[arg]
                        else:
                            break

                if len(params) == len(spec.args) - 1:
                    experiment[k] = clazz(**params)
                    configured.add(k)
                else:
                    unconfigured[k] = {arg for arg in spec.args[1:] if arg not in params}
            if configured:
                for k in configured:
                    del config[k]
            else:
                if config:
                    raise UnconfiguredItemsException(unconfigured) #TODO: prettify?
        return experiment
