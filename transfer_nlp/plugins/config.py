"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import inspect
import json
import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, Union, Any, Optional

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


class UnconfiguredItemsException(Exception):
    def __init__(self, items):
        super().__init__(f'There are some unconfigured items, which makes these items not configurable: {items}')
        self.items = items


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


def _replace(dico: Dict, env):
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
                v_upd = v_upd.replace('$' + env_key, env[env_key])

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


class ExperimentConfig2:

    def __init__(self, experiment: Union[str, Path, Dict], **env):
        """
        :param experiment: the experiment config
        :param env: substitution variables, e.g. a HOME directory. generally use all caps.
        :return: the experiment
        """
        self.factories: Dict[str, ConfigFactoryABC] = {}
        self.experiment: Dict[str, Any] = {}

        if isinstance(experiment, dict):
            config = dict(experiment)
        else:
            config = json.load(open(experiment))

        _replace(dico=config, env=env)

        # extract simple parameters
        logger.info(f"Initializing simple parameters:")
        experiment = {}
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

        self._build_items2(config)

    def recursive_build(self, object_key: str, object_dict: Dict):

        if '_name' not in object_dict:
            raise ValueError(f"The object a=should have a _name key to access its class")

        class_name = object_dict['_name']
        clazz = CLASSES.get(class_name)

        if not clazz:
            raise ValueError(
                f'The object class is named {object_dict["_name"]} but this name is not registered. see transfer_nlp.config.register_plugin for more information')

        spec = inspect.getfullargspec(clazz.__init__)
        params = {}
        param2config_key = {}
        named_params = {p: pv for p, pv in object_dict.items() if p != '_name'}
        default_params = {p: pv for p, pv in zip(reversed(spec.args), reversed(spec.defaults))} if spec.defaults else {}

        for arg in spec.args[1:]:

            if arg == 'experiment_config':
                params[arg] = self
                param2config_key[arg] = arg

            elif arg in named_params:
                value = named_params[arg]

                if isinstance(value, dict) and '_name' in value:
                    value = self.recursive_build(object_key=arg, object_dict=value)
                elif isinstance(value, str) and value[0] == '$' and value[1:] in self.experiment:
                    value = self.experiment[value[1:]]

                elif isinstance(value, dict) and '_name' not in value:
                    for item in value:
                        value[item] = self.recursive_build(object_key=item, object_dict=value[item])
                elif not isinstance(value, list) and value in self.experiment:
                    value = self.experiment[value]
                else:
                    pass
                params[arg] = value
                param2config_key[arg] = value

            # For values that are not in named_params, we look first at the experiment dict, then at the defaults parameters
            elif arg in self.experiment:
                params[arg] = self.experiment[arg]
                param2config_key[arg] = arg
            elif arg in default_params:
                value = default_params[arg]
                params[arg] = value
                param2config_key[arg] = None
            else:
                raise ValueError(f"{arg} is not a parameter from the {class_name} class")

        if object_key:
            self.factories[object_key] = PluginFactory(cls=clazz, param2config_key=param2config_key, **params)

        return clazz(**params)

    def _build_items2(self, config: Dict[str, Any]):

        while config:

            configured = set()

            for object_key, object_dict in config.items():

                try:
                    self.experiment[object_key] = self.recursive_build(object_key, object_dict)
                    configured.add(object_key)
                except Exception as e:
                    logger.info(f"Cannot configure the item '{object_key}' yet, we need to do another pass on the config file")

            if configured:
                for k in configured:
                    del config[k]
        logger.info("Experiment successfully configured!")



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


class ExperimentConfig:

    def __init__(self, experiment: Union[str, Path, Dict], **env):
        """
        :param experiment: the experiment config
        :param env: substitution variables, e.g. a HOME directory. generally use all caps.
        :return: the experiment
        """
        self.factories: Dict[str, ConfigFactoryABC] = {}
        self.experiment: Dict[str, Any] = None

        if isinstance(experiment, dict):
            config = dict(experiment)
        else:
            config = json.load(open(experiment))

        _replace(dico=config, env=env)

        # extract simple parameters
        logger.info(f"Initializing simple parameters:")
        experiment = {}
        for k, v in config.items():
            if not isinstance(v, dict) and not isinstance(v, list):
                logger.info(f"Parameter {k}: {v}")
                experiment[k] = v
                self.factories[k] = ParamFactory(v)

        # extract simple lists
        logger.info(f"Initializing simple lists:")
        for k, v in config.items():
            if isinstance(v, list) and all(not isinstance(vv, dict) and not isinstance(vv, list) for vv in v):
                logger.info(f"Parameter {k}: {v}")
                experiment[k] = v
                self.factories[k] = PluginFactory(list, None, v)

        for k in experiment:
            del config[k]

        try:
            logger.info(f"Initializing complex configurations ignoring default params:")
            self._build_items(config, experiment, 0)
        except UnconfiguredItemsException as e:
            pass

        try:
            logger.info(f"Initializing complex configurations only filling in default params not found in the experiment:")
            self._build_items(config, experiment, 1)
        except UnconfiguredItemsException as e:
            pass

        try:
            logger.info(f"Initializing complex configurations filling in all default params:")
            self._build_items(config, experiment, 2)
        except UnconfiguredItemsException as e:
            logging.error('There are unconfigured items in the experiment. Please check your configuration:')
            for k, v in e.items.items():
                logging.error(f'"{k}" missing properties:')
                for vv in v:
                    logging.error(f'\t+ {vv}')

            raise e

        self.experiment = experiment

    def _build_items(self, config: Dict[str, Any], experiment: Dict[str, Any], default_params_mode: int):
        """
        Build complex items
        In the config file, if we want to use a pre-built object as parameter for another object creation, we must use the $ specifier. e.g.:
        {
      "my_object": {
        "_name": "Foo",
        "param": "bar"
      },
      "my_second_object": {
        "_name": "SomeClass",
        "object_param": "$my_object"
      }
    }

        :param config:
        :param experiment:
        :param default_params_mode: 0 - ignore default params, 1 - only fill in default params not found in the experiment, 2 - fill in all default params
        :return: None
        :raise UnconfiguredItemsException: if items are unable to be configured
        """
        i = 1
        while config:
            logger.info(f"Pass {i}")
            i += 1
            configured = set()  # items configured in this iteration
            unconfigured = {}  # items unable to be configured in this iteration
            for k, v in config.items():
                logger.info(f"Parameter {k}")
                if not isinstance(v, dict):
                    raise ValueError(f'complex configuration object config[{k}] must be a dict')

                if '_name' not in v:
                    raise ValueError(f'complex configuration object config[{k}] must be have a "_name" property')

                clazz = CLASSES.get(v['_name'])
                if not clazz:
                    raise ValueError(
                        f'config[{k}] is named {v["_name"]} but this name is not registered. see transfer_nlp.config.register_plugin for more information')

                spec = inspect.getfullargspec(clazz.__init__)
                params = {}
                param2config_key = {}
                named_params = {p: pv for p, pv in v.items() if p != '_name'}
                default_params = {p: pv for p, pv in zip(reversed(spec.args), reversed(spec.defaults))} if spec.defaults else {}

                literal_params = {}
                for p, pv in v.items():

                    # Literals are any fixed values that are not string starting with '$'
                    if (not isinstance(pv, str) or not pv[0] == '$') and not isinstance(pv, list):
                        literal_params[p] = pv
                    if isinstance(pv, list) and all(not isinstance(item, str) or not item[0] == '$' for item in pv):
                        literal_params[p] = pv

                for arg in spec.args[1:]:

                    if arg in literal_params:
                        params[arg] = literal_params[arg]
                        param2config_key[arg] = None
                    else:
                        if arg == 'experiment_config':
                            params[arg] = self
                            param2config_key[arg] = arg
                        if arg in named_params:
                            alias = named_params[arg]
                            if isinstance(alias, list):
                                param_list = []
                                for p in alias:
                                    if p in experiment:
                                        param_list.append(experiment[p])
                                    if isinstance(p, str) and p[0] == '$' and p[1:] in experiment:
                                        param_list.append(experiment[p[1:]])
                                    else:
                                        break
                                if len(param_list) == len(alias):
                                    params[arg] = param_list
                            elif alias in experiment:
                                params[arg] = experiment[alias]
                            elif isinstance(alias, str) and alias[0] == '$' and alias[1:] in experiment:
                                params[arg] = experiment[alias[1:]]
                            param2config_key[arg] = alias
                        elif arg in experiment:
                            params[arg] = experiment[arg]
                            param2config_key[arg] = arg
                        elif default_params_mode == 1 and arg not in config and arg in default_params:
                            params[arg] = default_params[arg]
                            param2config_key[arg] = None
                        elif default_params_mode == 2 and arg in default_params:
                            params[arg] = default_params[arg]
                            param2config_key[arg] = None

                if len(params) == len(spec.args) - 1:
                    experiment[k] = clazz(**params)
                    self.factories[k] = PluginFactory(cls=clazz, param2config_key=param2config_key, **params)
                    configured.add(k)
                else:
                    unconfigured[k] = {arg for arg in spec.args[1:] if arg not in params}

            if configured:
                for k in configured:
                    del config[k]
            else:
                if config:
                    raise UnconfiguredItemsException(unconfigured)

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
