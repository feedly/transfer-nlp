import logging
from pathlib import Path

from experiments.bert.bert import *
from transfer_nlp.plugins.config import ExperimentConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    path = './bert.json'
    experiment = ExperimentConfig(path, HOME=home_env)
    experiment['trainer'].train()
