import logging
from pathlib import Path
from transfer_nlp.plugins.config import ExperimentConfig
from experiments.surnames import *
from experiments.news import *
from experiments.cbow import *

name = 'transfer_nlp.experiments.training'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    surname_paths = ['/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/mlp.json',
                     '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/surnamesRNN.json',
                     '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/surnameClassifier.json',
                     '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/surnamesGeneration.json'
                     ]
    cbow_path = '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/cbow.json'
    news_path = '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/experiments/newsClassifier.json'


    for path in surname_paths:
        logger.info(f"Launching test for experiment {path.split('/')[-1]}")
        experiment = ExperimentConfig.from_json(path, HOME=str(Path.home()))
        experiment['trainer'].train()
        if 'predictor' in experiment:
            input_json = {
                "inputs": ["Zhang",
                           "Mueller", 'Mahmoud', "Rastapopoulos"]}
            output_json = experiment['predictor'].json_to_json(input_json=input_json)
            logger.info(input_json)
            logger.info(output_json)


    logger.info(f"Launching test for experiment {cbow_path.split('/')[-1]}")
    path = cbow_path
    experiment = ExperimentConfig.from_json(path, HOME=str(Path.home()))
    experiment['trainer'].train()
    input_json = {
        "inputs": ["I go to and take notes"]}
    output_json = experiment['predictor'].json_to_json(input_json=input_json)
    logger.info(input_json)
    logger.info(output_json)


    logger.info(f"Launching test for experiment {news_path.split('/')[-1]}")
    path = news_path
    experiment = ExperimentConfig.from_json(path, HOME=str(Path.home()))
    experiment['trainer'].train()
    input_json = {
        "inputs": ["Banking financing Asset Manager Gets OK To Appeal €15M Fee Payout Ruling",
                   "NASA's New Planet-Hunting Telescope Just Found Its First Earth-Sized World"]}
    output_json = experiment['predictor'].json_to_json(input_json=input_json)
    logger.info(input_json)
    logger.info(output_json)
