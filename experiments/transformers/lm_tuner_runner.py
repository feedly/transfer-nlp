import logging
from pathlib import Path

from experiments.transformers.dataset import *
from experiments.transformers.model import *
from transfer_nlp.plugins.config import ExperimentConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    # # Train a language model on large dataset
    # experiment = ExperimentConfig('./lm_fine_tuning.json', HOME=home_env)
    # experiment.experiment['trainer'].train()

    # Fine-tune the LM on a classification task
    experiment = ExperimentConfig('./lm_clf_fine_tuning.json', HOME=home_env)
    experiment.experiment['trainer'].train()