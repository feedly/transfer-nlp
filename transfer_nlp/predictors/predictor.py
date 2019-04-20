import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union

import torch

from transfer_nlp.embeddings.embeddings import make_embedding_matrix
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.registry import Model, Data
from transfer_nlp.runners.runnersABC import _prepare_batch

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
            model_inputs = {inp: batch[inp] for inp in self.model.inputs_names}
            y_pred = self.model(**model_inputs)
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

    @classmethod
    def from_params(cls, experiment_file: Union[Path, str]) -> 'Predictor':
        """
        From the same experiment file used for training, load sufficient utilities to do inference
        :param experiment_file:
        :return:
        """
        experiments_path = Path(__file__).resolve().parent.parent
        experiments_path /= experiment_file

        with open(experiments_path, 'r') as exp:
            config_args = json.load(exp)

        config_args['vectorizer_file'] = config_args['save_dir'] + '/' + config_args['vectorizer_file']
        config_args['model_state_file'] = config_args['save_dir'] + '/' + config_args['model_state_file']

        vectorizer = Data.load_vectorizer_only(config_args=config_args)  # Improve design of Data to be able to do this
        config_args['num_embeddings'] = config_args['input_dim'] = len(vectorizer.data_vocab)
        config_args['initial_num_channels'] = len(vectorizer.data_vocab)
        config_args['output_dim'] = config_args['num_classes'] = len(vectorizer.target_vocab)
        if not torch.cuda.is_available():
            config_args['cuda'] = False
        config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

        # Word Embeddings
        if config_args.get('use_glove', False):
            words = vectorizer.data_vocab._token2id.keys()
            embeddings = make_embedding_matrix(glove_filepath=config_args['glove_filepath'],
                                               words=words)
            logging.info("Using pre-trained embeddings")
        else:
            logger.info("Not using pre-trained embeddings")
            embeddings = None

        # Register useful parameters and objects useful for model instantiation #TODO: do proper testing on this part
        config_args['pretrained_embeddings'] = embeddings

        if hasattr(vectorizer.data_vocab, 'mask_index'):
            config_args['padding_idx'] = config_args.get('padding_idx', 0)
        else:
            config_args['padding_idx'] = 0  # TODO: see if this fails

        model = Model.from_config(config_args=config_args)  # This should load the model object as well as pre-trained model weights

        return cls(model=model, vectorizer=vectorizer)


def register_predictor(predictor_class):
    """
    Register a predictor class into the Predictor registry
    :param predictor_class:
    :return:
    """
    existing_classes = Predictor.PREDICTOR_CLASSES
    if predictor_class.__name__ in existing_classes:
        existing = list(existing_classes.keys())
        raise ValueError(f"{predictor_class.__name__} is already registered. Please have a look at the existing predictors: {existing}")
    else:
        existing_classes[predictor_class.__name__] = predictor_class
        return predictor_class


@register_predictor
class MLPPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, model: Model, vectorizer: Vectorizer):
        super().__init__(model=model, vectorizer=vectorizer)

    def json_to_json(self, input_json: Dict):
        return {
            'x_in': torch.tensor([self.vectorizer.vectorize(input_string=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)
        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]


@register_predictor
class SurnameCNNPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, model: Model, vectorizer: Vectorizer):
        super().__init__(model=model, vectorizer=vectorizer)

    def json_to_data(self, input_json: Dict) -> Dict:
        return {
            'x_in': torch.Tensor([self.vectorizer.vectorize(input_string=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)
        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]


@register_predictor
class NewsPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, model: Model, vectorizer: Vectorizer):
        super().__init__(model=model, vectorizer=vectorizer)

    def json_to_data(self, input_json: Dict) -> Dict:
        vector_length = 30
        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(title=input_string, vector_length=vector_length) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)

        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]


@register_predictor
class SurnameRNNPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, model: Model, vectorizer: Vectorizer):
        super().__init__(model=model, vectorizer=vectorizer)

    def json_to_data(self, input_json: Dict) -> Dict:
        vector_length = 30

        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(surname=input_string, vector_length=vector_length)[0] for input_string in input_json['inputs']]),
            'x_lengths': torch.Tensor([self.vectorizer.vectorize(surname=input_string, vector_length=vector_length)[1] for input_string in input_json['inputs']])
        }

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output):
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)
        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]


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
