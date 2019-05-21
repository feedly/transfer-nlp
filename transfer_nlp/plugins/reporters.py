from abc import ABC
from pathlib import Path

from transfer_nlp.plugins.config import ExperimentConfig


class ReporterABC(ABC):
    """
    Reporter implementations write reports about trained models. They should at least produce human readable reports,
    but can additionally produce reports that are easily machine-parsable.
    """

    def report(self, experiment_name: str, experiment: ExperimentConfig, report_dir: Path) -> float:
        """
        report the results of an experiment
        :param experiment_name: the name of the experiment.
        :param experiment: the completed experiment.
        :param report_dir: the directory in which to write the report
        :return: the key metric value, it's assumed higher is better.
        """

        pass