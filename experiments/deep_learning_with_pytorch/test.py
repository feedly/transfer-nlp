import logging
from pathlib import Path

from transfer_nlp.plugins.config import ExperimentConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    examples = "experiments.deep_learning_with_pytorch.surnames"
    exp = "mlp2.yml"
    experiment = ExperimentConfig(exp,
                                  HOME=home_env,
                                  EXEMPLES=examples)
    experiment['trainer'].train()
