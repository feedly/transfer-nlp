from pathlib import Path
from unittest import TestCase

from transfer_nlp.runner.experiment_runner import load_config


class ConfigLoaderTest(TestCase):

    def test_loader(self):
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
