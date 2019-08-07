import configparser
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest import TestCase
from unittest.mock import patch

from transfer_nlp.plugins.config import register_plugin, ExperimentConfig
from transfer_nlp.plugins.reporters import ReporterABC
from transfer_nlp.plugins.trainers import TrainerABC
from transfer_nlp.runner.experiment_runner import ExperimentRunner, load_config


@register_plugin
class MockTrainer(TrainerABC):
    def __init__(self, bool_param, int_param, str_param, float_param, env_param):
        self.bool_param = bool_param
        self.int_param = int_param
        self.str_param = str_param
        self.float_param = float_param
        self.env_param = env_param
        self.trained = False

    def train(self):
        ExperimentRunnerTest._trainer_calls += 1
        if self.trained:
            raise ValueError()
        self.trained = True


@register_plugin
class MockReporter(ReporterABC):
    def __init__(self):
        self.reported = False

    def report(self, name: str, experiment: ExperimentConfig, report_dir: Path):
        ExperimentRunnerTest._configs[name] = experiment
        ExperimentRunnerTest._reporter_calls += 1
        if self.reported:
            raise ValueError()

        self.reported = True
        return ExperimentRunnerTest._reporter_calls

    @staticmethod
    def report_globally(aggregate_reports: Dict, report_dir: Path) -> Any:
        print("global reporting message")


class ExperimentRunnerTest(TestCase):
    _reporter_calls = 0
    _trainer_calls = 0
    _configs = {}

    def setUp(self):
        _reporter_calls = 0
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_run_all(self, mock_stdout):
        pkg_dir = Path(__file__).parent

        # Test the load_config works for both cfg and toml files
        cfg_config = load_config(p=pkg_dir / 'test_experiment.cfg')
        toml_config = load_config(p=pkg_dir / 'test_experiment.toml')
        self.assertIsInstance(cfg_config, dict)
        self.assertIsInstance(toml_config, dict)

        # Test that TOML is able to deal with lists, whereas cfg considers lists as a string
        # This is the main reason to prefere using TOML
        self.assertIsInstance(cfg_config["config1"]['lparam'], str)
        self.assertIsInstance(toml_config["config1"]['lparam'], list)

        experiment_cache = ExperimentRunner.run_all(experiment=pkg_dir / 'test_experiment.json',
                                                    experiment_config=pkg_dir / 'test_experiment.cfg',
                                                    report_dir=self.test_dir + '/reports',
                                                    trainer_config_name='the_trainer',
                                                    reporter_config_name='the_reporter', ENV_PARAM='my_env_param',
                                                    experiment_cache=pkg_dir / 'test_read_only.json')

        self.assertEqual(mock_stdout.getvalue(), "global reporting message\n")
        self.assertIsInstance(experiment_cache['another_trainer'], MockTrainer)
        self.assertEqual(experiment_cache['another_trainer'].int_param, 1)
        self.assertEqual(experiment_cache['another_trainer'].bool_param, True)

        self.assertEqual(2, ExperimentRunnerTest._reporter_calls)
        self.assertEqual(2, ExperimentRunnerTest._trainer_calls)

        self.assertEqual(2, len(ExperimentRunnerTest._configs))

        for name, bparam, iparam, fparam, sparam in [('config1', True, 1, 1.5, 'hello'), ('config2', False, 2, 2.5, 'world')]:
            # assert params where substituted into the experiment properly
            cfg = ExperimentRunnerTest._configs[name]['the_trainer']
            self.assertEqual(bparam, cfg.bool_param)
            self.assertEqual(iparam, cfg.int_param)
            self.assertEqual(fparam, cfg.float_param)
            self.assertEqual(sparam, cfg.str_param)
            self.assertEqual('my_env_param', cfg.env_param)

            # assert params were recorded in the reports directory
            cp = configparser.ConfigParser()
            cp.read(f'{self.test_dir}/reports/{name}/experiment.cfg')
            self.assertEqual(1, len(cp.sections()))
            self.assertEqual(bparam, cp.getboolean(name, 'bparam'))
            self.assertEqual(iparam, cp.getint(name, 'iparam'))
            self.assertEqual(fparam, cp.getfloat(name, 'fparam'))
            self.assertEqual(sparam, cp.get(name, 'sparam'))
            self.assertEqual('my_env_param', cp.get(name, 'ENV_PARAM'))

            self.assertEqual(ExperimentConfig.load_experiment_config(pkg_dir / 'test_experiment.json'),
                             ExperimentConfig.load_experiment_config(f'{self.test_dir}/reports/{name}/experiment.json'))
