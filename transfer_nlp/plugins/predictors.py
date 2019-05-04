import inspect
import logging
from itertools import zip_longest
from typing import Dict, List, Any

import torch
from ignite.utils import convert_tensor

from transfer_nlp.loaders.vectorizers import Vectorizer

logger = logging.getLogger(__name__)


def _prepare_batch(batch: Dict, device=None, non_blocking: bool = False):
    """Prepare batch for training: pass to a device with options.

    """
    result = {key: convert_tensor(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
    return result


class PredictorABC:

    def __init__(self, vectorizer: Vectorizer, model: torch.nn.Module):

        self.model: torch.nn.Module = model
        self.model.eval()
        self.forward_params = {}
        model_spec = inspect.getfullargspec(self.model.forward)
        for fparam, pdefault in zip_longest(reversed(model_spec.args[1:]), reversed(model_spec.defaults if model_spec.defaults else [])):
            self.forward_params[fparam] = pdefault

        self.vectorizer: Vectorizer = vectorizer

    def forward(self, batch: Dict[str, Any]) -> torch.tensor:
        """
        Do the forward pass
        :param batch:
        :return:
        """
        with torch.no_grad():
            batch = _prepare_batch(batch, device="cpu", non_blocking=False)
            model_inputs = {}
            for p, pdefault in self.forward_params.items():
                val = batch.get(p)
                if val is None:
                    if pdefault is None:
                        raise ValueError(f'missing model parameter "{p}"')
                    else:
                        val = pdefault

                model_inputs[p] = val
            y_pred = self.model(**model_inputs)

        return y_pred

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

    def predict(self, batch: Dict[str, Any]) -> List[Dict]:
        """
        Decode the output of the forward pass
        :param batch:
        :return:
        """
        forward = self.forward(batch=batch)
        return self.decode(forward)

    def json_to_json(self, input_json: Dict) -> Dict[str, Any]:
        """
        Full prediction: input_json --> data example --> predictions --> json output
        :param input_json:
        :return:
        """
        json2data = self.json_to_data(input_json=input_json)
        predictions = self.predict(batch=json2data)
        predictions2json = self.output_to_json(predictions)

        return predictions2json
