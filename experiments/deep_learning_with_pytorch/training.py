from pathlib import Path

from experiments.deep_learning_with_pytorch.surnames import *
from experiments.deep_learning_with_pytorch.cbow import *
from experiments.deep_learning_with_pytorch.news import *
from transfer_nlp.plugins.config import ExperimentConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    surname_paths = ['./deep_learning_with_pytorch/mlp.json',
                     './deep_learning_with_pytorch/surnamesRNN.json',
                     './deep_learning_with_pytorch/surnameClassifier.json',
                     './deep_learning_with_pytorch/surnamesGeneration.json'
                     ]
    cbow_path = './deep_learning_with_pytorch/cbow.json'
    news_path = './deep_learning_with_pytorch/newsClassifier.json'

    for path in surname_paths:
        logger.info(f"Launching test for experiment {path}")
        experiment = ExperimentConfig(path, HOME=home_env)
        experiment['trainer'].train()
        if 'predictor' in experiment:
            input_json = {
                "inputs": ["Zhang",
                           "Mueller", 'Mahmoud', "Rastapopoulos"]}
            output_json = experiment['predictor'].json_to_json(input_json=input_json)
            logger.info(input_json)
            logger.info(output_json)

    logger.info(f"Launching test for experiment {cbow_path}")
    path = cbow_path
    experiment = ExperimentConfig(path, HOME=home_env)
    experiment['trainer'].train()
    input_json = {
        "inputs": ["I go to and take notes"]}
    output_json = experiment['predictor'].json_to_json(input_json=input_json)
    logger.info(input_json)
    logger.info(output_json)

    logger.info(f"Launching test for experiment {news_path}")
    path = news_path
    experiment = ExperimentConfig(path, HOME=home_env)
    experiment['trainer'].train()
    input_json = {
        "inputs": ["Banking financing Asset Manager Gets OK To Appeal €15M Fee Payout Ruling",
                   "NASA's New Planet-Hunting Telescope Just Found Its First Earth-Sized World"]}
    output_json = experiment['predictor'].json_to_json(input_json=input_json)
    logger.info(input_json)
    logger.info(output_json)
