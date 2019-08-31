import configparser
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Union

import toml

from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.plugins.reporters import ReporterABC
from transfer_nlp.plugins.trainer_abc import TrainerABC

ConfigEnv = Dict[str, Any]


def load_config(p: Path) -> Dict[str, ConfigEnv]:
    p = Path(str(p)).expanduser()
    if p.suffix == '.toml':
        rv = toml.load(p)
        return rv

    if p.suffix != '.cfg':
        raise ValueError("Config files should be either .cfg or .toml files")

    def get_val(cfg: configparser.ConfigParser, section: str, key):
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
    def _capture_logs(report_path: Path):
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
    def run_all(experiment: Union[str, Path],
                experiment_config: Union[str, Path],
                report_dir: Union[str, Path],
                trainer_config_name: str = 'trainer',
                reporter_config_name: str = 'reporter',
                experiment_cache: Union[str, Path, Dict] = None,
                **env_vars) -> ExperimentConfig:
        """
        :param experiment: the experiment config
        :param experiment_config: the experiment config file. The cfg file should be defined in `ConfigParser
               <https://docs.python.org/3/library/configparser.html#module-configparser>`_ format such that
               each section is an experiment configuration.
        :param report_dir: the directory in which to produce the reports. It's recommended to include a timestamp your report directory so you
               can preserve previous reports across code changes. E.g. $HOME/reports/run_2019_02_22.
        :param trainer_config_name: the name of the trainer configuration object. The referenced object should implement `TrainerABC`.
        :param reporter_config_name: the name of the reporter configuration object. The referenced object should implement `ReporterABC`.
        :param experiment_cache: the experiment config with cached objects
        :param env_vars: any additional environment variables, like file system paths
        :return: the experiment cache
        """

        envs: Dict[str, ConfigEnv] = load_config(Path(experiment_config))

        report_path = Path(report_dir)
        report_path.mkdir(parents=True)

        # Before starting, save the 3 global files: experiment, configs and cache
        global_report_dir = report_path / 'global-reporting'
        global_report_dir.mkdir(parents=True)
        shutil.copy(src=str(experiment), dst=str(global_report_dir / str(Path(experiment).name)))
        shutil.copy(src=str(experiment), dst=str(global_report_dir / str(Path(experiment_cache).name)))
        shutil.copy(src=str(experiment_config), dst=str(global_report_dir / str(Path(experiment_config).name)))

        experiment_config_cache = {}
        if experiment_cache:
            logging.info("#" * 5 + f"Building a set of read-only objects and cache them for use in different experiment settings" + "#" * 5)
            experiment_config_cache = ExperimentConfig(experiment_cache, **env_vars)
            logging.info("#" * 5 + f"Read-only objects are built and cached for use in different experiment settings" + "#" * 5)

        aggregate_reports = {}
        for exp_name, env in envs.items():
            exp_report_path = report_path / exp_name
            exp_report_path.mkdir()
            log_handler = ExperimentRunner._capture_logs(exp_report_path)
            try:
                logging.info('running %s', exp_name)
                all_vars = dict(env_vars)
                all_vars.update(env)

                exp = deepcopy(experiment)
                if experiment_cache:
                    exp = ExperimentConfig.load_experiment_config(exp)
                    exp.update(experiment_config_cache)

                experiment_config = ExperimentConfig(exp, **all_vars)
                trainer: TrainerABC = experiment_config[trainer_config_name]
                reporter: ReporterABC = experiment_config[reporter_config_name]
                trainer.train()

                # Save the config for this particular experiment
                exp_config = {
                    exp_name: all_vars}
                with (exp_report_path / 'experiment_config.toml').open('w') as expfile:
                    toml.dump(exp_config, expfile)

                # Get this particular config reporting and store it in the
                # aggregated reportings
                report = reporter.report(exp_name, experiment_config, exp_report_path)
                aggregate_reports[exp_name] = report
            finally:
                ExperimentRunner._stop_log_capture(log_handler)

        reporter_class = experiment_config[reporter_config_name].__class__
        if issubclass(reporter_class, ReporterABC):
            reporter_class.report_globally(aggregate_reports=aggregate_reports, report_dir=global_report_dir)

        return experiment_config_cache
