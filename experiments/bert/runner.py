import logging
from pathlib import Path

from experiments.bert.bert import *
from transfer_nlp.plugins.config import ExperimentConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    path = './bert.json'
    experiment = ExperimentConfig.from_json(path, HOME=home_env)
    # Prepare optimizer

    layers = ['classifier.weight', 'classifier.bias']

    [parameter.requires_grad_(False) for name, parameter in experiment['trainer'].model.named_parameters() if name not in layers]
    param_optimizer = list(experiment['trainer'].model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if n in layers],
            'weight_decay': 0.01}
    ]

    experiment['trainer'].optimizer = BertAdam(optimizer_grouped_parameters, lr=experiment['lr'])
    experiment['trainer'].train()

