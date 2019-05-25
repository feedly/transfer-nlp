"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import inspect
import json
import logging
import os
from abc import abstractmethod, ABC
from enum import Enum
from pathlib import Path
from typing import Dict, Union, Any, Optional, AbstractSet, Set, List

import ignite.metrics as metrics
import torch.nn as nn
import torch.optim as optim
from smart_open import open

logger = logging.getLogger(__name__)

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
    "TanhShrink": nn.functional.tanhshrink,
    "Accuracy": metrics.Accuracy,
}


def register_plugin(clazz):
    if clazz.__name__ in CLASSES:
        raise ValueError(f"{clazz.__name__} is already registered to class {CLASSES[clazz.__name__]}. Please select another name")
    else:
        CLASSES[clazz.__name__] = clazz
        return clazz


class UnknownPluginException(Exception):
    def __init__(self, clazz: str):
        super().__init__(f'Class {clazz} is not registered. See transfer_nlp.config.register_plugin for more information.')
        self.clazz: str = clazz


class UnconfiguredItemsException(Exception):
    def __init__(self, items: Dict[str, Set]):
        super().__init__(f'There are some unconfigured items, which makes these items not configurable: {items}')
        self.items: Dict[str, Set] = items


class BadParameter(Exception):
    def __init__(self, clazz, param):
        super().__init__(f"Parameter naming error: '{param}' is not a parameter of class '{clazz}'")
        self.param = param
        self.clazz = clazz


class ConfigFactoryABC(ABC):

    @abstractmethod
    def create(self):
        pass


class ParamFactory(ConfigFactoryABC):
    """
    Factory for simple parameters
    """

    def __init__(self, param):
        self.param = param

    def create(self):
        return self.param


class PluginFactory(ConfigFactoryABC):
    """
    Factory for complex objects creation
    """

    def __init__(self, cls, param2config_key: Optional[Dict[str, str]], *args, **kwargs):
        self.cls = cls
        self.param2config_key = param2config_key
        self.args = args
        self.kwargs = kwargs

    def create(self):
        return self.cls(*self.args, **self.kwargs)


class DefaultParamsMode(Enum):
    IGNORE_DEFAULTS = 0,
    NOT_IN_EXPERIMENT = 1,
    USE_DEFAULTS = 2


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
                logger.info('*** updating parameter %s -> %s', v, v_upd)

        return v_upd

    def recursive_replace(dico: Dict):

        for k, v in dico.items():
            if not isinstance(v, dict) and not isinstance(v, list):
                v = do_env_subs(v)
                dico[k] = v
            elif isinstance(v, list) and all(not isinstance(vv, dict) and not isinstance(vv, list) for vv in v):
                upd = []
                for vv in v:
                    upd.append(do_env_subs(vv))
                    dico[k] = upd
            elif isinstance(v, dict):
                recursive_replace(dico[k])
            else:
                pass

    recursive_replace(dico=dico)


class ExperimentConfig:

    @staticmethod
    def load_experiment_json(experiment: Union[str, Path, Dict]) -> Dict:
        if isinstance(experiment, dict):
            config = dict(experiment)
        else:
            with open(experiment) as f:
                config = json.load(f)
        return config

    def __init__(self, experiment: Union[str, Path, Dict], **env):
        """
        :param experiment: the experiment config
        :param env: substitution variables, e.g. a HOME directory. generally use all caps.
        :return: the experiment
        """
        self.factories: Dict[str, ConfigFactoryABC] = {}
        self.experiment: Dict[str, Any] = {}

        config = ExperimentConfig.load_experiment_json(experiment)
        _replace_env_variables(dico=config, env=env)

        # extract simple parameters
        logger.info(f"Initializing simple parameters:")
        for k, v in config.items():
            if not isinstance(v, dict) and not isinstance(v, list):
                logger.info(f"Parameter {k}: {v}")
                self.experiment[k] = v
                self.factories[k] = ParamFactory(v)

        # extract simple lists
        logger.info(f"Initializing simple lists:")
        for k, v in config.items():
            if isinstance(v, list) and all(not isinstance(vv, dict) and not isinstance(vv, list) for vv in v):
                logger.info(f"Parameter {k}: {v}")
                self.experiment[k] = v
                self.factories[k] = PluginFactory(list, None, v)

        for k in self.experiment:
            del config[k]

        self._build_items(config)

    def _do_recursive_build(self, object_key: str, object_dict: Dict, default_params_mode: DefaultParamsMode, unconfigured_keys: AbstractSet,
                            parent_level: str):

        def resolve_simple_value(factory_key: str, val: Any) -> Any:
            if isinstance(val, str):
                if val[0] == '$':
                    keyval = val[1:]
                    if keyval in self.experiment:
                        self.factories[factory_key] = self.factories[keyval]
                        return self.experiment[keyval]
                    else:
                        raise UnconfiguredItemsException({factory_key: {val}})

            return val

        def do_recursive_build_list(list_object: List, arg_name: str) -> List:
            """
            Function to recursively deal with nested lists
            :param list_object:
            :param arg_name:
            :return:
            """
            copy = []
            for i, element in enumerate(list_object):
                if isinstance(element, dict):
                    if "_name" in element:
                        element = self._do_recursive_build(object_key=str(i), object_dict=element,
                                                           default_params_mode=default_params_mode,
                                                           unconfigured_keys=unconfigured_keys,
                                                           parent_level=f'{parent_level}.{arg_name}.{i}')
                    else:
                        for key in element:
                            if isinstance(element[key], dict):
                                element[key] = self._do_recursive_build(object_key=key, object_dict=element[key],
                                                                       default_params_mode=default_params_mode,
                                                                       unconfigured_keys=unconfigured_keys,
                                                                       parent_level=f'{parent_level}.{arg}.{i}.{key}')

                            elif isinstance(element[key], list):
                                element[key] = do_recursive_build_list(list_object=element[key], arg_name=f"{arg_name}.{i}.{key}")
                            else:
                                element[key] = resolve_simple_value(f'{parent_level}.{arg}.{key}', element[key])
                        self.factories[f'{parent_level}.{arg}'] = PluginFactory(dict, None, list(element.items()))

                elif isinstance(element, list):
                    element = do_recursive_build_list(list_object=element, arg_name=f"{arg_name}.{i}")

                else:
                    element = resolve_simple_value(f'{parent_level}.{arg_name}.{i}', element)
                copy.append(element)

            result = copy
            self.factories[f'{parent_level}.{arg_name}'] = PluginFactory(list, None, list(copy))
            return result

        logger.info(f"Configuring {object_key}")

        if '_name' not in object_dict:
            raise ValueError(f"The object {object_key} should have a _name key to access its class")

        class_name = object_dict['_name']
        clazz = CLASSES.get(class_name)

        if not clazz:
            raise UnknownPluginException(object_dict["_name"])

        if inspect.isclass(clazz):
            spec = inspect.getfullargspec(clazz.__init__)
            spec_args = spec.args[1:]
        elif inspect.isfunction(clazz):
            spec = inspect.getfullargspec(clazz)
            spec_args = spec.args
        elif inspect.ismethod(clazz):
            spec = inspect.getfullargspec(clazz)
            spec_args = spec.args[1:]
        else:
            raise ValueError(f"{class_name} should be either a class, a function or a method")

        params = {}
        param2config_key = {}
        named_params = {p: pv for p, pv in object_dict.items() if p != '_name'}
        default_params = {p: pv for p, pv in zip(reversed(spec.args), reversed(spec.defaults))} if spec.defaults else {}

        for named_param in named_params:
            if named_param not in spec_args:
                raise BadParameter(clazz=class_name, param=named_param)

        for arg in spec_args:

            if arg == 'experiment_config':
                params[arg] = self
                param2config_key[arg] = arg

            elif arg in named_params:
                value = named_params[arg]

                if isinstance(value, dict):
                    if '_name' in value:
                        value = self._do_recursive_build(object_key=arg, object_dict=value,
                                                         default_params_mode=default_params_mode,
                                                         unconfigured_keys=unconfigured_keys,
                                                         parent_level=parent_level + "." + arg)
                    else:
                        for item in value:
                            if isinstance(value[item], dict):
                                value[item] = self._do_recursive_build(object_key=item, object_dict=value[item],
                                                                       default_params_mode=default_params_mode,
                                                                       unconfigured_keys=unconfigured_keys,
                                                                       parent_level=f'{parent_level}.{arg}.{item}')
                            else:  # value[item] is either an object defined in a dictionary, or it's an already built object
                                value[item] = resolve_simple_value(f'{parent_level}.{arg}.{item}', value[item])
                        self.factories[f'{parent_level}.{arg}'] = PluginFactory(dict, None, list(value.items()))

                elif isinstance(value, list):
                    value = do_recursive_build_list(list_object=value, arg_name=arg)

                elif isinstance(value, str) and value[0] == '$':

                    if value[1:] in self.experiment:
                        logger.info(f"Using the object {value}, already instantiated")
                        value = self.experiment[value[1:]]
                    else:
                        logger.info(f"{value} not configured yet, will be configured in next iteration")
                else:
                    logger.info(f"Using value {arg} / {named_params[arg]} from the config file")

                if not (isinstance(value, str) and value[0] == '$'):
                    params[arg] = value
                    param2config_key[arg] = value
                else:
                    logger.debug(f"You need to define '{value}' as a '{value[1:]}' object in the config file")

            # For values that are not in named_params, we look first at the experiment dict, then at the defaults parameters
            elif arg in self.experiment:
                params[arg] = self.experiment[arg]
                param2config_key[arg] = arg

            elif default_params_mode == DefaultParamsMode.NOT_IN_EXPERIMENT and \
                    arg not in self.experiment and arg not in unconfigured_keys and \
                    arg in default_params:
                params[arg] = default_params[arg]
                param2config_key[arg] = None
            elif default_params_mode == DefaultParamsMode.USE_DEFAULTS and arg in default_params:
                params[arg] = default_params[arg]
                param2config_key[arg] = None

        if len(params) == len(spec_args):
            self.factories[parent_level] = PluginFactory(cls=clazz, param2config_key=param2config_key, **params)
            return clazz(**params)

        else:
            unconfigured_params = spec_args - params.keys()
            raise UnconfiguredItemsException({parent_level: unconfigured_params})

    def _build_items_with_default_params_mode(self, config: Dict, default_params_mode: DefaultParamsMode):

        while config:
            configured = set()
            config_errors = {}
            for object_key, object_dict in config.items():

                try:
                    self.experiment[object_key] = self._do_recursive_build(object_key, object_dict,
                                                                           default_params_mode=default_params_mode,
                                                                           unconfigured_keys=config.keys(),
                                                                           parent_level=object_key)
                    configured.add(object_key)

                except BadParameter as b:
                    raise BadParameter(clazz=b.clazz, param=b.param)
                except UnconfiguredItemsException as e:
                    config_errors.update(e.items)
            if configured:
                for k in configured:
                    del config[k]

            else:
                if config_errors:
                    raise UnconfiguredItemsException(config_errors)

    def _build_items(self, config: Dict[str, Any]):

        try:
            logger.info(f"Initializing complex configurations ignoring default params:")
            self._build_items_with_default_params_mode(config, DefaultParamsMode.IGNORE_DEFAULTS)
        except UnconfiguredItemsException as e:
            pass

        try:
            logger.info(f"Initializing complex configurations only filling in default params not found in the experiment:")

            self._build_items_with_default_params_mode(config, DefaultParamsMode.NOT_IN_EXPERIMENT)
        except UnconfiguredItemsException as e:
            pass

        try:
            logger.info(f"Initializing complex configurations filling in all default params:")
            self._build_items_with_default_params_mode(config, DefaultParamsMode.USE_DEFAULTS)
        except UnconfiguredItemsException as e:
            logging.error('There are unconfigured items in the experiment. Please check your configuration:')
            for k, v in e.items.items():
                logging.error(f'"{k}" missing properties:')
                for vv in v:
                    logging.error(f'\t+ {vv}')

            raise e

    def _check_init(self):
        if self.experiment is None:
            raise ValueError('experiment config is not setup yet!')

    # map-like methods
    def __getitem__(self, item):
        self._check_init()
        return self.experiment[item]

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
