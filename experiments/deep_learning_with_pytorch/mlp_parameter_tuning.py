from datetime import datetime
from pathlib import Path

from smart_open import open

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
    logging.basicConfig(level=logging.INFO)
    parent_dir = Path(__file__).parent
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    date = '_'.join(str(datetime.today()).split(' '))

    # # Uncomment to run the sequential Runner without caching read-only objects
    # ExperimentRunner.run_all(experiment=parent_dir / 'mlp_parameter_tuning.json',
    #                          experiment_config=parent_dir / 'mlp_parameter_tuning.cfg',
    #                          report_dir=f"{home_env}/mlp_parameter_fine_tuning/{date}",
    #                          trainer_config_name='trainer',
    #                          reporter_config_name='reporter', HOME=home_env)


    # # Uncomment to run the sequential Runner with caching read-only objects
    # ExperimentRunner.run_all(experiment=parent_dir / 'mlp_parameter_tuning_uncached.json',
    #                          experiment_config=parent_dir / 'mlp_parameter_tuning.cfg',
    #                          report_dir=f"{home_env}/mlp_parameter_fine_tuning/{date}",
    #                          trainer_config_name='trainer',
    #                          reporter_config_name='reporter',
    #                          experiment_cache=parent_dir / 'mlp_parameter_tuning_cache.json',
    #                          HOME=home_env)
