from pathlib import Path
from experiments.deep_learning_with_pytorch.surnames import *

from transfer_nlp.plugins.config import register_plugin, ExperimentConfig
from transfer_nlp.plugins.reporters import ReporterABC
from transfer_nlp.runner.experiment_runner import ExperimentRunner


@register_plugin
class MyReporter(ReporterABC):
    def __init__(self):
        self.reported = False

    def report(self, name: str, experiment: ExperimentConfig, report_dir: Path):

        with open(report_dir / 'metrics_report.txt', 'w') as reporting:
            reporting.write(f"Metrics reporting for experiment {name}\n")
            reporting.write("#"*50 + '\n')

            for mode, metrics in experiment['trainer'].metrics_history.items():
                reporting.write(f"Reporting metrics in {mode} mode\n")
                for metric, values in metrics.items():
                    reporting.write(f"{metric}: [{', '.join([str(value) for value in values])}]\n")


if __name__ == "__main__":

    dir = Path(__file__).parent
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    ExperimentRunner.run_all(experiment=dir / 'mlp_parameter_tuning.json',
                             experiment_config=dir / 'mlp_parameter_tuning.cfg',
                             report_dir=dir / 'reports',
                             trainer_config_name='trainer',
                             reporter_config_name='reporter', HOME=home_env)