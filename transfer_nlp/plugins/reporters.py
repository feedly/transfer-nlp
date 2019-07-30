from abc import ABC
from pathlib import Path
from typing import Any, Dict

from transfer_nlp.plugins.config import ExperimentConfig


class ReporterABC(ABC):
    """
    Reporter implementations write reports about trained models. They should at least produce human readable reports,
    but can additionally produce reports that are easily machine-parsable.
    """

    def report(self, experiment_name: str, experiment: ExperimentConfig, report_dir: Path) -> Any:
        """
        report the results of an experiment
        :param experiment_name: the name of the experiment.
        :param experiment: the completed experiment.
        :param report_dir: the directory in which to write the report
        :return: the key metric value, it's assumed higher is better.
        """

        pass

    @staticmethod
    def report_globally(aggregate_reports: Dict, report_dir: Path) -> Any:
        """
        do a global reporting for multiple experiment configurations
        :param aggregate_reports: the result of report() on each experiment config.
        :param report_dir: the directory in which to write the report
        :return: a global reporting of the key metric values along different configurations.
        """
        pass
