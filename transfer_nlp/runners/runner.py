# import logging
from pathlib import Path
from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.experiments.surnames import SurnameDatasetGeneration, SurnameConditionedGenerationModel, SurnameVectorizerGeneration1
from transfer_nlp.predictors.predictor import *
from transfer_nlp.experiments.news import *
from transfer_nlp.experiments.cbow import *
from transfer_nlp.models.cnn import *
from transfer_nlp.embeddings.embeddings import Embedding, EmbeddingsHyperParams

name = 'transfer_nlp.runners.runner'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    path = '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/transfer_nlp/experiments/surnamesGeneration2.json'
    experiment = ExperimentConfig.from_json(path, HOME=str(Path.home()))
    experiment['trainer'].train()
    # input_json = {
    #     "inputs": ["Zhang",
    #                "Mueller", 'Mahmoud', "Rastapopoulos"]}
    # output_json = experiment['predictor'].json_to_json(input_json=input_json)
    # logger.info(input_json)
    # logger.info(output_json)

    # input_json = {
    #     "inputs": ["Banking financing Asset Manager Gets OK To Appeal €15M Fee Payout Ruling",
    #                "NASA's New Planet-Hunting Telescope Just Found Its First Earth-Sized World"]}
    # output_json = experiment['predictor'].json_to_json(input_json=input_json)
    #
    # logger.info(input_json)
    # logger.info(output_json)

    # input_json = {
    #     "inputs": ["I go to and take notes"]}
    # output_json = experiment['predictor'].json_to_json(input_json=input_json)
    #
    # logger.info(input_json)
    # logger.info(output_json)
