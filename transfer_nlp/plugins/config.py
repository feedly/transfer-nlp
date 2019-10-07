"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...
The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import logging
import os
import traceback
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Type, Union

import toml
import yaml

logger = logging.getLogger(__name__)
REGISTRY = {}


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


class InstantiationError(Exception):
    """
    An error happened while instantiating an experiment
    """

    def __init__(self, obj_name: str):
        """
        :param obj_name: The name of the object that couldn't be instantiated
        """
        self.obj_name: str = obj_name

    def __str__(self) -> str:
        return f"Failed to instantiate `{self.obj_name}`"


class CallableInstantiationError(InstantiationError):
    """
    An error happened while instantiating a callable
    """

    def __init__(self, obj_name: str, callable_name: str, arg_names: List[str]):
        """
        :param obj_name: The name of the object that couldn't be instantiated 
        :param callable_name: The callable's name that failed when called
        :param arg_names: The names of the arguments that were passed to the callable
        """
        super().__init__(obj_name)
        self.callable_name: str = callable_name
        self.arg_names: List[str] = arg_names

    def __str__(self) -> str:
        return f"{super().__str__()} calling `{self.callable_name}` using the arguments " + ', '.join(self.arg_names) + '; see exception raised above'


class LoopInConfigError(InstantiationError):
    """
    An object refers to itself
    """

    def __str__(self) -> str:
        return f"{super().__str__()} because it refers to itself"


class UnknownPluginException(InstantiationError):
    """
    A registrable has not been registred
    """
    def __init__(self, object_name: str, registrable: str):
        super().__init__(object_name)
        self.registrable: str = registrable

    def __str__(self) -> str:
        return f"{super().__str__()} because registrable `{self.registrable}` isn't registred"


class UnknownReferenceError(InstantiationError):
    """
    An object that is referenced does not exist
    """
    def __init__(self, object_name: str, reference_name: str):
        super().__init__(object_name)
        self.reference_name: str = reference_name

    def __str__(self) -> str:
        return f"{super().__str__()} because it reference to `{self.reference_name}` that doesn't exist"


class InstantiationImpossible(Exception):
    pass


class ObjectInstantiator(metaclass=ABCMeta):
    """
    An instantiator that knows how to instantiate 1 kind of object
    """

    def __init__(self):
        self.builder: ObjectBuilder = None

    def set_builder(self, builder: "ObjectBuilder") -> None:
        """
        Set the builder, for recursive puposes

        :param builder: The main builder to use when recursivity is needed
        :return: None
        """
        self.builder: ObjectBuilder = builder

    @abstractmethod
    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        """
        :param config: The config from which we want to instantiate the object
        :param name: The full name of the object to instantiate
        :return: The instantiated object
        """
        raise InstantiationImpossible


class ObjectBuilder:
    """
    The main builder class
    """
    def __init__(self, instantiators: List[ObjectInstantiator]):
        """
        :param instantiators: The ordered list of instantiators that can be used to instantiate the objects
        """
        self.instantiators: List[ObjectInstantiator] = instantiators
        for instantiator in instantiators:
            instantiator.set_builder(self)

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        """
        build the object trying to use all the instantiators in a row, until one works

        :param config: The config of the object to instantiate
        :param name: The full name of this object
        :return: The instantiated object
        """

        for instantiator in self.instantiators:
            try:
                return instantiator.instantiate(config, name)
            except InstantiationImpossible:
                pass

        logger.info(f'instantiating "{name}" as a simple object, {config}')

        return config


class DictInstantiator(ObjectInstantiator):
    """
    Instantiate a dictionary
    """

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Dict:
        """
        :param config: The config for the object to instantiate
        :param name: The full name of the object we try to instantiate
        :return: The resulting object
        :raise InstantiationImpossible: If the instantiator cannot handle this config
        """
        if not isinstance(config, dict):
            raise InstantiationImpossible()

        logger.info(f'instantiating "{name}" as a dictionary')

        return {
            key: self.builder.instantiate(value_config, f'{name}.{key}')
            for key, value_config in config.items()
        }


class ListInstantiator(ObjectInstantiator):
    """
    Instantiate a list
    """

    def instantiate(self, config: Union[Dict, str, List], name: str) -> List:
        """
        :param config: The config for the object to instantiate
        :param name: The full name of the object we try to instantiate
        :return: The resulting object
        :raise InstantiationImpossible: If the instantiator cannot handle this config
        """
        if not isinstance(config, list):
            raise InstantiationImpossible()

        logger.info(f'instantiating "{name}" as a list')

        return [
            self.builder.instantiate(value_config, f'{name}.{i}')
            for i, value_config in enumerate(config)
        ]


class FromMappingInstantiator(ObjectInstantiator):
    """
    Instantiate an object looking for its reference in a Mapping object
    """

    def __init__(self, mapping: Mapping[str, Any], mapping_name: str):
        """
        :param mapping: The mapping to use to look for references 
        :param mapping_name: The mapping name, for logs
        """
        self.env: Mapping[str, Any] = mapping
        self.mapping_name: str = mapping_name

        super().__init__()

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        """
        :param config: The config for the object to instantiate
        :param name: The full name of the object we try to instantiate
        :return: The resulting object
        :raise InstantiationImpossible: If the instantiator cannot handle this config
        """

        if not isinstance(config, str) or not config.startswith('$'):
            raise InstantiationImpossible
        try:
            instance = self.env[config[1:]]
            logger.info(f'instantiating "{name}" using value {instance} from key {config} in {self.mapping_name}')
            return instance
        except KeyError:
            raise InstantiationImpossible


class FromEnvironmentVariableInstantiator(FromMappingInstantiator):
    """
    Instantiate an object from the environment variables
    """

    def __init__(self, env: Dict[str, Any]):
        """
        :param env: The dictionary of the environment variables
        """
        super().__init__(env, 'Environment')

        self.strings_to_replace: List[str, str] = [
            key
            for key, value in self.env.items()
            if isinstance(value, str) or isinstance(value, os.PathLike)
        ]
        self.strings_to_replace.sort(key=len, reverse=True)

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        """
        :param config: The config for the object to instantiate
        :param name: The full name of the object we try to instantiate
        :return: The resulting object
        :raise InstantiationImpossible: If the instantiator cannot handle this config
        :raise UnknownReferenceError: If the config refers to an unknown object
        """
        try:
            return self.builder.instantiate(super().instantiate(config, name), f'{name}')
        except InstantiationImpossible:
            if not isinstance(config, str):
                raise InstantiationImpossible()

            v_upd: str = config
            for key in self.strings_to_replace:
                v_upd = v_upd.replace(f'${key}', str(self.env[key]))

            if v_upd.startswith('$'):
                raise UnknownReferenceError(name, config) from None

            logger.info(f'instantiating "{name}" using value {v_upd}')

            return v_upd


class CallableInstantiator(DictInstantiator):
    """
    Instantiate an object calling a registered callable
    """

    def instantiate(self, config: Union[Dict, str, List], name: str) -> Any:
        """
        :param config: The config for the object to instantiate
        :param name: The full name of the object we try to instantiate
        :return: The instantiated object
        :raise InstantiationImpossible: If the instantiator cannot handle this config
        :raise UnknownPluginException: If the callable is not registered
        :raise CallableInstantiationError: If an error occurred while calling the callable
        """
        if not isinstance(config, dict) or not '_name' in config:
            raise InstantiationImpossible()

        klass_name: str = config['_name']

        if klass_name not in REGISTRY:
            raise UnknownPluginException(object_name=name, registrable=klass_name)

        klass: Union[Type, Callable] = REGISTRY[klass_name]

        logger.info(f'instantiating "{name}" calling {klass_name}')

        param_instances: Dict[str, Any] = {
            key: self.builder.instantiate(value_config, f'{name}.{key}')
            for key, value_config in config.items()
            if key != '_name'
        }

        try:
            return klass(**param_instances)
        except Exception:
            raise CallableInstantiationError(obj_name=name, callable_name=klass_name, arg_names=list(param_instances.keys()))


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

        self.builds_started: List[str] = []
        self.builders = [
            CallableInstantiator(),
            DictInstantiator(),
            ListInstantiator(),
            FromMappingInstantiator(REGISTRY, 'Registry'),
            FromMappingInstantiator(self, 'Experiment objects'),
            FromEnvironmentVariableInstantiator(env),
        ]

        self.builder: ObjectBuilder = ObjectBuilder(self.builders)

        self.experiment: Dict[str, Any] = {}

        for key, value_config in self.config.items():
            if key not in self.experiment:
                self.build(key)

    def _check_init(self):
        if self.experiment is None:
            raise ValueError('experiment config is not setup yet!')

    def build(self, key: str) -> Any:
        if key not in self.config:
            raise KeyError()
        if key in self.builds_started:
            raise LoopInConfigError(key)
        self.builds_started.append(key)
        self.experiment[key] = self.builder.instantiate(self.config[key], name=key)
        return self.experiment[key]

    # map-like methods
    def __getitem__(self, item):
        self._check_init()
        if item in self.experiment:
            return self.experiment[item]
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
