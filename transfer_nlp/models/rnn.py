import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transfer_nlp.common.utils import describe
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.registry import register_model

name = 'transfer_nlp.models.rnn'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


def column_gather(y_out: torch.FloatTensor, x_lengths: torch.LongTensor) -> torch.FloatTensor:

    x_lengths = x_lengths.long().detach().cpu().numpy() - 1

    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])

    return torch.stack(out)


@register_model
class ElmanRNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool=False):

        super(ElmanRNN, self).__init__()

        self.rnn_cell: nn.RNNCell = nn.RNNCell(input_size, hidden_size)

        self.batch_first: bool = batch_first
        self.hidden_size: int = hidden_size

    def _initial_hidden(self, batch_size: int) -> torch.tensor:
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in: torch.Tensor, initial_hidden: torch.Tensor=None) -> torch.Tensor:
        """

        :param x_in: an input data tensor.
                If self.batch_first: x_in.shape = (batch, seq_size, feat_size)
                Else: x_in.shape = (seq_size, batch, feat_size)
        :param initial_hidden: the initial hidden state for the RNN
        :return: The outputs of the RNN at each time step.
                If self.batch_first: hiddens.shape = (batch, seq_size, hidden_size)
                Else: hiddens.shape = (seq_size, batch, hidden_size)
        """

        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()

        hiddens = []

        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        hidden_t = initial_hidden

        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)

        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)

        return hiddens


@register_model
class SurnameClassifierRNN(nn.Module):

    def __init__(self, embedding_size: int, num_embeddings: int, num_classes: int,
                 rnn_hidden_size: int, batch_first: bool=True, padding_idx: int=0):

        super(SurnameClassifierRNN, self).__init__()

        self.emb: nn.Embedding = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=padding_idx)
        self.rnn: ElmanRNN = ElmanRNN(input_size=embedding_size,
                             hidden_size=rnn_hidden_size,
                             batch_first=batch_first)
        self.fc1: nn.Linear = nn.Linear(in_features=rnn_hidden_size,
                         out_features=rnn_hidden_size)
        self.fc2: nn.Linear = nn.Linear(in_features=rnn_hidden_size,
                          out_features=num_classes)

    def forward(self, x_in: torch.Tensor, x_lengths: torch.Tensor=None, apply_softmax: bool=False) -> torch.Tensor:
        """

        :param x_in: an input data tensor.
                 x_in.shape should be (batch, input_dim)
        :param x_lengths: the lengths of each sequence in the batch.
                 They are used to find the final vector of each sequence
        :param apply_softmax: a flag for the softmax activation
                 should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, output_dim)
        """

        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)

        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]

        y_out = F.relu(self.fc1(F.dropout(y_out, 0.5)))
        y_out = self.fc2(F.dropout(y_out, 0.5))

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out


def predict_nationalityRNN(surname: str, classifier: SurnameClassifierRNN, vectorizer: Vectorizer) -> Dict[str, Any]:

    vectorized_surname, vec_length = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
    vec_length = torch.tensor([vec_length], dtype=torch.int64)

    result = classifier(vectorized_surname, vec_length, apply_softmax=True)
    probability_values, indices = result.max(dim=1)

    index = indices.item()
    prob_value = probability_values.item()

    predicted_nationality = vectorizer.target_vocab.lookup_index(index)

    return {
        'nationality': predicted_nationality,
        'probability': prob_value,
        'surname': surname}


if __name__ == "__main__":

    input_size = 10
    hidden_size = 10
    seq_size = 50
    batch_size = 32

    model = ElmanRNN(input_size=input_size, hidden_size=hidden_size)

    tensor = torch.randn(size=(batch_size, seq_size, input_size))
    describe(tensor)
    output = model(x_in=tensor)
    describe(output)


    embedding_size = 100
    num_embeddings = 100
    num_classes = 10
    rnn_hidden_size = 64

    model = SurnameClassifierRNN(embedding_size=embedding_size, num_embeddings=num_embeddings, num_classes=num_classes,
                 rnn_hidden_size=rnn_hidden_size)

    tensor = torch.randint(low=1, high=num_embeddings, size=(batch_size, embedding_size))
    lens = torch.randint(low=1, high=num_embeddings, size=(batch_size,))
    describe(tensor)
    describe(lens)
    output = model(x_in=tensor, x_lengths=lens)
    describe(output)



