"""
This file contains all necessary plugins classes that the framework will use to let a user interact with custom models, data loaders, etc...

The Registry pattern used here is inspired from this post: https://realpython.com/primer-on-python-decorators/
"""
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, Union, Any

import torch.nn as nn
import torch.optim as optim
import ignite.metrics as metrics
from smart_open import open

name = 'transfer_nlp.plugins.registry'
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
        super().__init__('unconfigured items')
        self.items = items

class ExperimentConfig:

    @staticmethod
    def from_json(experiment:Union[str, Path, Dict], **env) -> Dict[str, Any]:
        """
        :param experiment: the experiment config
        :param env: substitution variables, e.g. a HOME directory. generally use all caps.
        :return: the experiment
        """

        env_keys = sorted(env.keys(), key=lambda k: len(k), reverse=True)

        def do_env_subs(v:Any) -> str:
            v_upd = v
            if isinstance(v_upd, str):
                for env_key in env_keys:
                    v_upd = v_upd.replace(env_key, env[env_key])

                if v_upd != v:
                    logger.info('*** updating parameter %s -> %s', v, v_upd)

            return v_upd

        if isinstance(experiment, dict):
            config = dict(experiment)
        else:
            config = json.load(open(experiment))

        #extract simple parameters
        experiment = {}
        for k,v in config.items():
            if not isinstance(v, dict) and not isinstance(v, list):
                experiment[k] = do_env_subs(v)

        #extract simple lists
        for k,v in config.items():
            if isinstance(v, list) and all(not isinstance(vv, dict) and not isinstance(vv, list) for vv in v):
                upd = []
                for vv in v:
                    upd.append(do_env_subs(vv))
                experiment[k] = upd

        for k in experiment:
            del config[k]

        try:
            ExperimentConfig._build_items(config, experiment, 0)
        except UnconfiguredItemsException as e:
            pass

        try:
            ExperimentConfig._build_items(config, experiment, 1)
        except UnconfiguredItemsException as e:
            pass


        try:
            ExperimentConfig._build_items(config, experiment, 2)
        except UnconfiguredItemsException as e:
            logging.error('There are unconfigured items in the experiment. Please check your configuration:')
            for k,v in e.items.items():
                logging.error(f'"{k}" missing properties:')
                for vv in v:
                    logging.error(f'\t+ {vv}')

            raise e

        return experiment

    @staticmethod
    def _build_items(config: Dict[str, Any], experiment: Dict[str, Any], default_params_mode: int):
        """

        :param config:
        :param experiment:
        :param default_params_mode: 0 - ignore default params, 1 - only fill in default params not found in the experiment, 2 - fill in all default params
        :return: None
        :raise UnconfiguredItemsException: if items are unable to be configured
        """

        while config:
            configured = set()  # items configured in this iteration
            unconfigured = {}   # items unable to be configured in this iteration
            for k, v in config.items():
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
                default_params = {p: pv for p, pv in zip(reversed(spec.args), reversed(spec.defaults))} if spec.defaults else {}

                literal_params = {}
                for p, pv in v.items():
                    if p[-1] == '_':
                        literal_params[p[:-1]] = pv
                    elif isinstance(pv, list):
                        for pvv in pv:
                            if not isinstance(pvv, str):
                                raise ValueError(f'string required for parameter names in list paramters...use key_ notation "{p}_" if you want to specify a literal parameter values.')

                    elif not isinstance(pv, str):
                        raise ValueError(f'string required for parameter names...use key_ notation "{p}_" if you want to specify a literal parameter value.')

                for arg in spec.args[1:]:
                    if arg in literal_params:
                        params[arg] = literal_params[arg]
                    else:
                        if arg in named_params:
                            alias = named_params[arg]
                            if isinstance(alias, list):
                                param_list = []
                                for p in alias:
                                    if p in experiment:
                                        param_list.append(experiment[p])
                                    else:
                                        break
                                if len(param_list) == len(alias):
                                    params[arg] = param_list
                            elif alias in experiment:
                                params[arg] = experiment[alias]
                        elif arg in experiment:
                            params[arg] = experiment[arg]
                        elif default_params_mode == 1 and arg not in config and arg in default_params:
                            params[arg] = default_params[arg]
                        elif default_params_mode == 2 and arg in default_params:
                            params[arg] = default_params[arg]

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
                    raise UnconfiguredItemsException(unconfigured)
