"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Type, Union

import ignite.metrics as metrics
import toml
import torch.nn as nn
import torch.optim as optim
import yaml


logger = logging.getLogger(__name__)

REGISTRY = {
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
    "TanhShrink": nn.functional.tanhshrink,
    "Accuracy": metrics.Accuracy,
}


def register_plugin(registrable: Any, alias: str = None):
    """
    Register a class, a function or a method to REGISTRY
    Args:
        registrable: 
        alias: 

    Returns:

    """
    if not alias:
        if registrable.__name__ in REGISTRY:
            raise ValueError(f"{registrable.__name__} is already registered to registrable {REGISTRY[registrable.__name__]}. Please select another name")
        else:
            REGISTRY[registrable.__name__] = registrable
            return registrable
    else:
        if alias in REGISTRY:
            raise ValueError(f"{alias} is already registered to registrable {REGISTRY[alias]}. Please select another name")
        else:
            REGISTRY[alias] = registrable
            return registrable


class UnknownPluginException(Exception):
    def __init__(self, registrable: str):
        super().__init__(f'Registrable object {registrable} is not registered. See transfer_nlp.config.register_plugin for more information.')
        self.registrable: str = registrable


class UnconfiguredItemsException(Exception):
    def __init__(self, items: Dict[str, Set]):
        super().__init__(f'There are some unconfigured items, which makes these items not configurable: {items}')
        self.items: Dict[str, Set] = items


class InstantiationImpossible(Exception):
    pass


class ObjectInstantiator(metaclass=ABCMeta):

    def __init__(self):
        self.builder: ObjectBuilder = None

    def set_builder(self, builder: "ObjectBuilder"):
        self.builder: ObjectBuilder = builder

    @abstractmethod
    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        raise InstantiationImpossible


class ObjectBuilder:
    def __init__(self, instantiators: List[ObjectInstantiator]):
        self.instantiators: List[ObjectInstantiator] = instantiators
        for instantiator in instantiators:
            instantiator.set_builder(self)

    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        for instanciator in self.instantiators:
            try:
                return instanciator.instantiate(config)
            except InstantiationImpossible:
                pass

        return config


class DictInstantiator(ObjectInstantiator):

    def instantiate(self, config: Union[Dict, str, List]) -> Dict:
        if not isinstance(config, dict):
            raise InstantiationImpossible()
        return {
            key: self.builder.instantiate(value_config)
            for key, value_config in config.items()
        }


class ListInstantiator(ObjectInstantiator):

    def instantiate(self, config: Union[Dict, str, List]) -> List:
        if not isinstance(config, dict):
            raise InstantiationImpossible()
        return [
            self.builder.instantiate(value_config)
            for value_config in config
        ]


class ENVInstantiator(ObjectInstantiator):

    def __init__(self, env: Dict[str, Any]):
        self.env: Dict[str, Any] = env
        super().__init__()

    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        if not isinstance(config, str) or not config.startswith('$'):
            raise InstantiationImpossible
        try:
            return self.env[config[1:]]
        except KeyError:
            raise InstantiationImpossible


class RegistryInstantiator(ObjectInstantiator):

    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        if not isinstance(config, str) or not config.startswith('$'):
            raise InstantiationImpossible
        try:
            return REGISTRY[config[1:]]
        except KeyError:
            raise InstantiationImpossible


class MainObjectInstantiator(ObjectInstantiator):

    def __init__(self, experiment: "ExperimentConfig"):
        self.experiment: ExperimentConfig = experiment
        super().__init__()

    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        if not isinstance(config, str) or not config.startswith('$'):
            raise InstantiationImpossible
        try:
            return self.experiment[config[1:]]
        except KeyError:
            raise InstantiationImpossible


class ClassInstantiator(DictInstantiator):

    def instantiate(self, config: Union[Dict, str, List]) -> Any:
        if not isinstance(config, dict) or not '_name' in config:
            raise InstantiationImpossible()

        klass: Union[Type, Callable] = REGISTRY[config['_name']]

        param_instances: Dict[str, Any] = super().instantiate({name: c for name, c in config.items() if name != '_name'})

        return klass(**param_instances)


class ExperimentConfig:

    @staticmethod
    def load_experiment_config(experiment: Union[str, Path, Dict]) -> Dict:
        config = {}
        if isinstance(experiment, dict):
            config = dict(experiment)
        else:
            experiment_path = Path(str(experiment)).expanduser()
            with experiment_path.open() as f:
                if experiment_path.suffix in {'.json', '.yaml', '.yml'}:
                    config = yaml.safe_load(f)
                elif experiment_path.suffix in {'.toml'}:
                    config = toml.load(f)
                else:
                    raise ValueError("Only Dict, json, yaml and toml experiment files are supported")
        return config

    def __init__(self, experiment: Union[str, Path, Dict], **env):
        """
        :param experiment: the experiment config
        :param env: substitution variables, e.g. a HOME directory. generally use all caps.
        :return: the experiment
        """

        self.config: Dict[str, Any] = ExperimentConfig.load_experiment_config(experiment)
        self.builds_started: List[str] = []

        self.builder: ObjectBuilder = ObjectBuilder([
            ClassInstantiator(),
            DictInstantiator(),
            ENVInstantiator(env),
            RegistryInstantiator(),
            MainObjectInstantiator(self),
        ])

        self.experiment: Dict[str, Any] = {}

        for key, value_config in self.config.items():
            if key not in self.experiment:
                self.build(key)

    def _check_init(self):
        if self.experiment is None:
            raise ValueError('experiment config is not setup yet!')

    def build(self, key: str) -> Any:
        if key in self.builds_started:
            raise Exception('Loop in config')
        self.builds_started.append(key)
        self.experiment[key] = self.builder.instantiate(self.config[key])
        return self.experiment[key]

    # map-like methods
    def __getitem__(self, item):
        self._check_init()
        try:
            return self.experiment[item]
        except KeyError:
            return self.build(item)

    def get(self, item, default=None):
        self._check_init()
        return self.experiment.get(item, default)

    def __iter__(self):
        self._check_init()
        return iter(self.experiment)

    def items(self):
        self._check_init()
        return self.experiment.items()

    def values(self):
        self._check_init()
        return self.experiment.values()

    def keys(self):
        self._check_init()
        return self.experiment.keys()

    def __setitem__(self, key, value):
        raise ValueError("cannot update experiment!")


@register_plugin
class A:
    def __init__(self, a: int, b: int = 2, c: int = 3):
        print(a, b, c)

exp = ExperimentConfig(
    {
        'test': 'coucou',
        'third': '$second',
        'second': '$VAR',
        'a': {
            '_name': 'A',
            'a': 4,
            'c': 5
        },
    },
    VAR=5,
)

print(exp.experiment)
