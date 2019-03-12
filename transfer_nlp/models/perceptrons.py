import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders.vectorizers import Vectorizer

name = 'transfer_nlp'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class Perceptron(nn.Module):

    def __init__(self, num_features):

        super(Perceptron, self).__init__()
        self.fc = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:

        y_out = self.fc(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)

        return y_out


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # TODO: experiment with more layers

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        # TODO: experiment with other activation functions

        intermediate = F.relu(self.fc(x_in))
        output = self.fc2(intermediate)

        if self.output_dim == 1:
            output = output.squeeze()

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output


def preprocess(text: str) -> str:
    """
    Basic text preprocessing
    :param text:
    :return:
    """

    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)

    return text


def predict_review(review: str, model: nn.Module, vectorizer: Vectorizer, threshold: float = 0.5):
    """
    Do inference from a text review
    :param review:
    :param classifier:
    :param vectorizer:
    :param threshold:
    :return:
    """

    review = preprocess(review)
    vector = torch.tensor(vectorizer.vectorize(review=review))
    classifier = model.cpu()
    result = classifier(vector.view(1, -1)).unsqueeze(dim=0)

    if len(result.size()) == 1:  # if y_pred contains binary logits, then just compute the sigmoid to get probas
        result = (torch.sigmoid(result) > threshold).cpu().long().item()  # .max(dim=1)[1]
    else:  # then we are in the softmax case, and we take the max
        _, result = result.max(dim=1)

    return vectorizer.target_vocab.lookup_index(index=result)


def inspect_model(model: nn.Module, vectorizer: Vectorizer):
    """
    Check the extreme words (positives and negatives) for linear models
    :param classifier:
    :param vectorizer:
    :return:
    """

    fc_weights = model.fc.weight.detach()[0]
    _, indices = torch.sort(fc_weights, dim=0, descending=True)
    indices = indices.numpy().tolist()

    logger.info("#"*50)
    logger.info("Top positive words:")
    logger.info("#"*50)
    for i in range(20):
        logger.info(vectorizer.data_vocab.lookup_index(index=indices[i]))

    logger.info("#"*50)
    logger.info("Top negative words:")
    logger.info("#"*50)
    indices.reverse()
    for i in range(20):
        logger.info(vectorizer.data_vocab.lookup_index(index=indices[i]))

