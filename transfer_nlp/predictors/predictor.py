import json
import logging
from pathlib import Path
from typing import Dict, List

import torch

from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.registry import Model, Data
from transfer_nlp.runners.runnersABC import _prepare_batch
from transfer_nlp.loaders.loaders import SurnamesDataset

name = 'transfer_nlp.predictors.predictor'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class Predictor:
    PREDICTOR_CLASSES = {}

    def __init__(self, model: torch.nn.Module, vectorizer: Vectorizer):
        self.model: torch.nn.Module = model
        self.model.eval()
        self.vectorizer: Vectorizer = vectorizer

    def json_to_data(self, input_json: Dict):
        """
        Transform a json entry into a data exemple
        :param input_json:
        :return:
        """
        raise NotImplementedError

    def output_to_json(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def infer(self, batch):
        """
        Use the model to infer on some data example
        :param args:
        :param kwargs:
        :return:
        """
        with torch.no_grad():
            batch = _prepare_batch(batch, device="cpu", non_blocking=False)
            model_inputs = {inp: batch[inp] for inp in self.model.inputs_names}
            y_pred = self.model(**model_inputs)
        return y_pred

    def predict(self, input_json: Dict):
        """
        Do inference and then decode the result
        :param args:
        :param kwargs:
        :return:
        """

        return self.decode(output=self.infer(self.json_to_data(input_json=input_json)))

    def json_to_json(self, input_json: Dict):
        """
        Full prediction: input_json --> data example --> inference result --> decoded result --> json ouput
        :param input_json:
        :return:
        """

        return self.output_to_json(self.predict(input_json=input_json))

    @classmethod
    def from_params(cls, experiment_file: Path):
        experiments_path = Path(__file__).resolve().parent.parent
        experiments_path /= experiment_file

        with open(experiments_path, 'r') as exp:
            config_args = json.load(exp)

        config_args['vectorizer_file'] = config_args['save_dir'] + '/' + config_args['vectorizer_file']
        config_args['model_state_file'] = config_args['save_dir'] + '/' + config_args['model_state_file']

        vectorizer = Data.load_vectorizer_only(config_args=config_args)  # Improve design of Data to be able to do this
        config_args['input_dim'] = len(vectorizer.data_vocab)
        config_args['output_dim'] = len(vectorizer.target_vocab)
        config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")
        model = Model.from_config(config_args=config_args)  # This should load the model object as well as pre-trained model weights

        return cls(model=model, vectorizer=vectorizer)


def register_predictor(predictor_class):
    existing_classes = Predictor.PREDICTOR_CLASSES
    if predictor_class.__name__ in existing_classes:
        existing = list(existing_classes.keys())
        raise ValueError(f"{predictor_class.__name__} is already registered. Please have a look at the existing predictors: {existing}")
    else:
        existing_classes[predictor_class.__name__] = predictor_class
        return predictor_class


@register_predictor
class MyPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, model: Model, vectorizer: Vectorizer):
        super().__init__(model=model, vectorizer=vectorizer)

    def json_to_data(self, input_json: Dict):
        return {
            'x_in': torch.Tensor([self.vectorizer.vectorize(input_string=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List):
        return {
            "outputs": outputs}

    def decode(self, output):
        print(output.size())
        _, result = output.max(dim=1)
        result = [int(res) for res in result]

        return [self.vectorizer.target_vocab.lookup_index(index=res) for res in result]


if __name__ == "__main__":
    experiment_file = "experiments/mlp.json"

    predictor = MyPredictor.from_params(experiment_file=experiment_file)
    input_json = {
        "inputs": ["Zhang", "Mueller"]}
    output_json = predictor.json_to_json(input_json=input_json)

    logger.info(input_json)
    logger.info(output_json)
