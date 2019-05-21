import configparser
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Union

from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.plugins.reporters import ReporterABC
from transfer_nlp.plugins.trainers import TrainerABC

ConfigEnv = Dict[str, Any]

def load_config(p: Path) -> Dict[str, ConfigEnv]:

    def get_val(cfg:configparser.ConfigParser, section: str, key):
        try:
            return cfg.getint(section, key)
        except ValueError:
            pass
        try:
            return cfg.getfloat(section, key)
        except ValueError:
            pass
        try:
            return cfg.getboolean(section, key)
        except ValueError:
            pass

        return cfg[section][key]

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(p)

    rv = {}

    for exp_name in cfg.sections():
        exp = {}
        for key in cfg[exp_name].keys():
            exp[key] = get_val(cfg, exp_name, key)
        rv[exp_name] = exp

    return rv

class ExperimentRunner:
    """
    Run an experiment several times with varying configurations.

    This class facilitates reusing a single json experiment file across several different configuations.
    """

    @staticmethod
    def _capture_logs(report_path:Path):
        logger = logging.getLogger('')
        handler = logging.FileHandler(str(report_path / 'runner.log'))
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')  # TODO configurable?
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        return handler

    @staticmethod
    def _stop_log_capture(handler):
        logger = logging.getLogger('')
        logger.removeHandler(handler)

    @staticmethod
    def _write_config(cfg_name: str, experiment:Dict, cfg:ConfigEnv, exp_report_path:Path):
        """duplicate the config used to run the experiment in the report directory to preserve history"""
        config = configparser.ConfigParser({}, OrderedDict)
        config.optionxform = str
        config.add_section(cfg_name)
        for k in sorted(cfg.keys()):
            config.set(cfg_name, k, str(cfg[k]))

        with (exp_report_path / 'experiment.cfg').open('w') as configfile:
            config.write(configfile)

        with (exp_report_path / 'experiment.json').open('w') as expfile:
            json.dump(experiment, expfile, indent=4)

    @staticmethod
    def run_all(experiment: Union[str, Path, Dict],
                experiment_config: Union[str, Path],
                report_dir: Union[str, Path],
                trainer_config_name: str = 'trainer',
                reporter_config_name: str = 'reporter',
                **env_vars) -> None:
        """
        :param experiment: the experiment config
        :param experiment_config: the experiment config file. The cfg file should be defined in `ConfigParser
               <https://docs.python.org/3/library/configparser.html#module-configparser>`_ format such that
               each section is an experiment configuration.
        :param report_dir: the directory in which to produce the reports. It's recommended to include a timestamp your report directory so you
               can preserve previous reports across code changes. E.g. $HOME/reports/run_2019_02_22.
        :param trainer_config_name: the name of the trainer configuration object. The referenced object should implement `TrainerABC`.
        :param reporter_config_name: the name of the reporter configuration object. The referenced object should implement `ReporterABC`.
        :param env_vars: any additional environment variables, like file system paths
        :return: None
        """

        envs: Dict[str, ConfigEnv] = load_config(Path(experiment_config))

        report_path = Path(report_dir)
        report_path.mkdir(parents=True)

        for exp_name, env in envs.items():
            exp_report_path = report_path / exp_name
            exp_report_path.mkdir()
            log_handler = ExperimentRunner._capture_logs(exp_report_path)
            try:
                logging.info('running %s', exp_name)
                all_vars = dict(env_vars)
                all_vars.update(env)
                experiment_config = ExperimentConfig(experiment, **all_vars)
                trainer: TrainerABC = experiment_config[trainer_config_name]
                reporter: ReporterABC = experiment_config[reporter_config_name]
                trainer.train()
                exp_json = ExperimentConfig.load_experiment_json(experiment)
                ExperimentRunner._write_config(exp_name, exp_json, all_vars, exp_report_path)
                reporter.report(exp_name, experiment_config, exp_report_path)
            finally:
                ExperimentRunner._stop_log_capture(log_handler)
