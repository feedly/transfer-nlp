import logging
from pathlib import Path

from experiments.bert.bert import *
from transfer_nlp.plugins.config import ExperimentConfig, ExperimentConfig2

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    path = './bert.json'
    experiment = ExperimentConfig2(path, HOME=home_env)
    experiment.experiment['trainer'].train()
