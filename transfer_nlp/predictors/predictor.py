import inspect
import logging
from itertools import zip_longest
from typing import Dict, List, Any

import torch

# from transfer_nlp.embeddings.embeddings import make_embedding_matrix
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.plugins.trainers import BasicTrainer
from transfer_nlp.runners.runnersABC import _prepare_batch

name = 'transfer_nlp.predictors.predictor'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


@register_plugin
class PredictorHyperParams(ObjectHyperParams):

    def __init__(self, trainer: BasicTrainer):
        super().__init__()
        self.vectorizer = trainer.dataset_splits.vectorizer
        self.model = trainer.model


class Predictor:

    def __init__(self, predictor_hyper_params: PredictorHyperParams):

        self.model: torch.nn.Module = predictor_hyper_params.model
        self.model.eval()
        self.forward_params = {}
        model_spec = inspect.getfullargspec(self.model.forward)
        for fparam, pdefault in zip_longest(reversed(model_spec.args[1:]), reversed(model_spec.defaults if model_spec.defaults else [])):
            self.forward_params[fparam] = pdefault

        self.vectorizer: Vectorizer = predictor_hyper_params.vectorizer

    def _forward(self, batch):
        model_inputs = {}
        for p, pdefault in self.forward_params.items():
            val = batch.get(p)
            if val is None:
                if pdefault is None:
                    raise ValueError(f'missing model parameter "{p}"')
                else:
                    val = pdefault

            model_inputs[p] = val

        return self.model(**model_inputs)

    def json_to_data(self, input_json: Dict) -> Dict:
        """
        Transform a json entry into a data example, which is the same that what the __getitem__ method in the
        data loader, except that this does not output any expected label as in supervised setting
        :param input_json:
        :return:
        """
        raise NotImplementedError

    def output_to_json(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Convert the result into a proper json
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def decode(self, *args, **kwargs) -> List[Dict]:
        """
        Return an output dictionary for every example in the batch
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def infer(self, batch: Dict[str, Any]) -> torch.tensor:
        """
        Use the model to infer on some data example
        :param batch:
        :return:
        """
        with torch.no_grad():
            batch = _prepare_batch(batch, device="cpu", non_blocking=False)
            y_pred = self._forward(batch)
        return y_pred

    def json_to_json(self, input_json: Dict) -> Dict[str, Any]:
        """
        Full prediction: input_json --> data example --> inference result --> decoded result --> json ouput
        :param input_json:
        :return:
        """

        json2data = self.json_to_data(input_json=input_json)
        data2infer = self.infer(json2data)
        infer2decode = self.decode(output=data2infer)
        decode2json = self.output_to_json(infer2decode)

        return decode2json












if __name__ == "__main__":
    experiment_file = "experiments/mlp.json"

    predictor = MLPPredictor.from_params(experiment_file=experiment_file)
    input_json = {
        "inputs": ["Zhang", "Mueller"]}
    output_json = predictor.json_to_json(input_json=input_json)

    logger.info(input_json)
    logger.info(output_json)

    experiment_file = "experiments/surnameClassifier.json"
    predictor = SurnameCNNPredictor.from_params(experiment_file=experiment_file)
    input_json = {
        "inputs": ["Zhang", "Mueller"]}
    output_json = predictor.json_to_json(input_json=input_json)

    logger.info(input_json)
    logger.info(output_json)

    experiment_file = "experiments/newsClassifier.json"
    predictor = NewsPredictor.from_params(experiment_file=experiment_file)

    input_json = {
        "inputs": ["Asset Manager Gets OK To Appeal €15M Fee Payout Ruling",
                   "NASA's New Planet-Hunting Telescope Just Found Its First Earth-Sized World"]}
    output_json = predictor.json_to_json(input_json=input_json)

    logger.info(input_json)
    logger.info(output_json)

    experiment_file = "experiments/surnamesRNN.json"
    predictor = SurnameRNNPredictor.from_params(experiment_file=experiment_file)

    input_json = {
        "inputs": ["Zhang",
                   "Mueller", 'Mahmoud', "Rastapopoulos"]}
    output_json = predictor.json_to_json(input_json=input_json)

    logger.info(input_json)
    logger.info(output_json)
