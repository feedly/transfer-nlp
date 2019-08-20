"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...
The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union, Mapping

import ignite.metrics as metrics
import toml
import torch.nn as nn
import torch.optim as optim
import yaml

logger = logging.getLogger(__name__)
REGISTRY = {}
# REGISTRY = {
#     'CrossEntropyLoss': nn.CrossEntropyLoss,
#     'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
#     "Adam": optim.Adam,
#     "SGD": optim.SGD,
#     "AdaDelta": optim.Adadelta,
#     "AdaGrad": optim.Adagrad,
#     "SparseAdam": optim.SparseAdam,
#     "AdaMax": optim.Adamax,
#     "ASGD": optim.ASGD,
#     "LBFGS": optim.LBFGS,
#     "RMSPROP": optim.RMSprop,
#     "Rprop": optim.Rprop,
#     "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
#     "MultiStepLR": optim.lr_scheduler.MultiStepLR,
#     "ExponentialLR": optim.lr_scheduler.ExponentialLR,
#     "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
#     "LambdaLR": optim.lr_scheduler.LambdaLR,
#     "ReLU": nn.functional.relu,
#     "LeakyReLU": nn.functional.leaky_relu,
#     "Tanh": nn.functional.tanh,
#     "Softsign": nn.functional.softsign,
#     "Softshrink": nn.functional.softshrink,
#     "Softplus": nn.functional.softplus,
#     "Sigmoid": nn.Sigmoid,
#     "CELU": nn.CELU,
#     "SELU": nn.functional.selu,
#     "RReLU": nn.functional.rrelu,
#     "ReLU6": nn.functional.relu6,
#     "PReLU": nn.functional.prelu,
#     "LogSigmoid": nn.functional.logsigmoid,
#     "Hardtanh": nn.functional.hardtanh,
#     "Hardshrink": nn.functional.hardshrink,
#     "ELU": nn.functional.elu,
#     "Softmin": nn.functional.softmin,
#     "Softmax": nn.functional.softmax,
#     "LogSoftmax": nn.functional.log_softmax,
#     "GLU": nn.functional.glu,
#     "TanhShrink": nn.functional.tanhshrink,
#     "Accuracy": metrics.Accuracy,
# }


def register_plugin(registrable: Any, alias: str = None):
    """
    Register a class, a function or a method to REGISTRY
    Args:
        registrable:
        alias:
    Returns:
    """
    alias = alias or registrable.__name__

    if alias in REGISTRY:
        raise ValueError(f"{alias} is already registered to registrable {REGISTRY[alias]}. Please select another name")

    REGISTRY[alias] = registrable
    return registrable


class CallableInstantiationError(Exception):
    pass


class CallableInstantiation(Exception):
    def __init__(self, name: str, klass_name: str):
        super().__init__(f'Error happened while instantiating "{name}", calling {klass_name}')
        self.name: str = name
        self.klass_name = klass_name


class UnknownPluginException(CallableInstantiationError):
    def __init__(self, registrable: str):
        super().__init__(f'Registrable object {registrable} is not registered. See transfer_nlp.config.register_plugin for more information.')
        self.registrable: str = registrable


class InstantiationImpossible(Exception):
    pass


class ObjectInstantiator(metaclass=ABCMeta):

    def __init__(self):
        self.builder: ObjectBuilder = None

    def set_builder(self, builder: "ObjectBuilder"):
        self.builder: ObjectBuilder = builder

    @abstractmethod
    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        raise InstantiationImpossible


class ObjectBuilder:
    def __init__(self, instantiators: List[ObjectInstantiator]):
        self.instantiators: List[ObjectInstantiator] = instantiators
        for instantiator in instantiators:
            instantiator.set_builder(self)

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:

        for instantiator in self.instantiators:
            try:
                return instantiator.instantiate(config, name)
            except InstantiationImpossible:
                pass

        logging.info(f'instantiating "{name}" as a simple object, {config}')

        return config


class DictInstantiator(ObjectInstantiator):

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Dict:
        if not isinstance(config, dict):
            raise InstantiationImpossible()

        logging.info(f'instantiating "{name}" as a dictionary')

        return {
            key: self.builder.instantiate(value_config, f'{name}.{key}')
            for key, value_config in config.items()
        }


class ListInstantiator(ObjectInstantiator):

    def instantiate(self, config: Union[Dict, str, List], name: str) -> List:
        if not isinstance(config, list):
            raise InstantiationImpossible()

        logging.info(f'instantiating "{name}" as a list')

        return [
            self.builder.instantiate(value_config, f'{name}.{i}')
            for i, value_config in enumerate(config)
        ]

import os
class FromMappingInstantiator(ObjectInstantiator):

    def __init__(self, env: Mapping[str, Any], mapping_name: str):
        self.env: Mapping[str, Any] = env
        self.mapping_name: str = mapping_name

        super().__init__()

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:

        if not isinstance(config, str) or not config.startswith('$'):
            raise InstantiationImpossible
        try:
            instance = self.env[config[1:]]
            logging.info(f'instantiating "{name}" using value {instance} from key {config} in {self.mapping_name}')
            return instance
        except KeyError:
            # if config[1:] == 'bar':
            #     print('error')
            raise InstantiationImpossible


class CallableInstantiator(DictInstantiator):

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        if not isinstance(config, dict) or not '_name' in config:
            raise InstantiationImpossible()

        klass_name: str = config['_name']

        if klass_name not in REGISTRY:
            raise UnknownPluginException(klass_name)

        klass: Union[Type, Callable] = REGISTRY[klass_name]

        logging.info(f'instantiating "{name}" calling {klass_name}')

        param_instances: Dict[str, Any] = {
            key: self.builder.instantiate(value_config, f'{name}.{key}')
            for key, value_config in config.items()
            if key != '_name'
        }

        try:
            return klass(**param_instances)
        except Exception:
            raise CallableInstantiation(name=name, klass_name=klass_name)


def _replace_env_variables(dico: Dict, env: Dict) -> None:
    """
    Replace all occurrences of environment variable to particular strings
    :param dico:
    :param env:
    :return:
    """
    env_keys = sorted(env.keys(), key=lambda k: len(k), reverse=True)

    def do_env_subs(v: Any) -> str:
        v_upd = v
        if isinstance(v_upd, str):
            for env_key in env_keys:
                env_val = env[env_key]

                if isinstance(env_val, os.PathLike):
                    env_val = str(env_val)

                if not isinstance(env_val, str):
                    if v == f'${env_key}':
                        # allow for non string value replacements
                        v_upd = env_val
                        break
                else:
                    v_upd = v_upd.replace('$' + env_key, env_val)

            if v_upd != v:
                logger.debug('*** updating parameter %s -> %s', v, v_upd)

        return v_upd

    def recursive_replace(my_item: Union[Dict, List, str]):

        if isinstance(my_item, dict):
            for k, v in my_item.items():
                if not isinstance(v, dict) and not isinstance(v, list):
                    v = do_env_subs(v)
                    my_item[k] = v

                elif isinstance(v, dict):
                    recursive_replace(my_item[k])
                elif isinstance(v, list):
                    recursive_replace(my_item[k])
                else:
                    pass
        elif isinstance(my_item, list):
            for i, item in enumerate(my_item):
                if not isinstance(item, list) and not isinstance(item, dict):
                    my_item[i] = do_env_subs(item)
                else:
                    recursive_replace(item)

    recursive_replace(my_item=dico)


class ExperimentConfig(Mapping[str, Any]):

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
        _replace_env_variables(dico=self.config, env=env)
        self.builds_started: List[str] = []

        self.builder: ObjectBuilder = ObjectBuilder([
            CallableInstantiator(),
            DictInstantiator(),
            ListInstantiator(),
            FromMappingInstantiator(env, 'Environment'),
            FromMappingInstantiator(REGISTRY, 'Registry'),
            FromMappingInstantiator(self, 'Experiment objects'),
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
        self.experiment[key] = self.builder.instantiate(self.config[key], name=key)
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

    def __len__(self) -> int:
        return len(self.experiment)


@register_plugin
class A:
    def __init__(self, a: int, b: int = 2, c: int = 3):
        pass

    @staticmethod
    def g(**kwargs):
        return kwargs


register_plugin(A.g, alias='A.g')


@register_plugin
def f(a: int, b: int = 2, **kwargs):
    c = sum([v for k, v in kwargs.items()])
    return a, b, c


logging.basicConfig(level='INFO')

exp = ExperimentConfig(
    {
        'test': 'coucou',
        'third': '$second',
        'second': [
            '$VAR',
            '$test'
        ],
        'a': {
            '_name': 'A',
            'a': 4,
            'c': 5,
        },
        'None': {
            '_name': 'f',
            'a': 5,
            'b': 5,
            'r': 2,
            'd': 10
        },
        'DictObject': {
            "first_key": {
                '_name': 'f',
                'a': 5,
                'b': 5,
                'r': 2,
                'd': 10
            },
            "list_key": [
                1,
                2,
                "$VAR",
                "$PATH/some/path"
            ]
        }
    },
    VAR=10,
    PATH='/tmp'
)

# print(exp.experiment)
