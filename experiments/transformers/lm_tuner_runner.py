import logging
from pathlib import Path

from experiments.transformers.model import *
from transfer_nlp.plugins.config import ExperimentConfig
from ..utils import PLUGINS

logger = logging.getLogger(__name__)

for plugin_name, plugin in PLUGINS.items():
    register_plugin(registrable=plugin, alias=plugin_name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    # # Train a language model on large dataset
    # experiment = ExperimentConfig('./lm_fine_tuning.json', HOME=home_env)
    # experiment.experiment['trainer'].train()

    # # Fine-tune the LM on a classification task
    # experiment = ExperimentConfig('./lm_clf_fine_tuning.json', HOME=home_env)
    # experiment.experiment['trainer'].train()

    # # Fine-tune the LM on a classification task with an adapted Transformer
    # # You can change the trainers param "adaptation" to experiment with several
    # # adaptation schemes
    # experiment = ExperimentConfig('./lm_clf_adaptation.json', HOME=home_env)
    # experiment.experiment['trainer'].train()

    # Train jointly the LM and the classifier, branched on the same
    # transformer backbone
    experiment = ExperimentConfig('./multitask.json', HOME=home_env)
    experiment.experiment['trainer'].train()
