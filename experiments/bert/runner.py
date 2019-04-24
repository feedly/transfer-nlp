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
    param_optimizer = list(experiment['trainer'].model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    experiment['trainer'].optimizer = BertAdam(optimizer_grouped_parameters,
                                               lr=experiment['lr'])

    import inspect

    lines = inspect.getsource(experiment['trainer'].model.forward)
    print(lines)
    experiment['trainer'].train()
