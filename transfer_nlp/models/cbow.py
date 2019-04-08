import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from transfer_nlp.common.utils import describe
from transfer_nlp.plugins.registry import register_model

name = 'transfer_nlp.models.cbow'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


@register_model
class CBOWClassifier(nn.Module):  # Simplified cbow Model

    def __init__(self, vocabulary_size: int, embedding_size: int, padding_idx: int=0):

        super(CBOWClassifier, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=vocabulary_size)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) -> torch.Tensor:
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """

        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out


if __name__ == "__main__":

    vocabulary_size = 5000
    embedding_size = 100
    seq_size = 50
    batch_size = 32

    model = CBOWClassifier(vocabulary_size=vocabulary_size, embedding_size=embedding_size)

    tensor = torch.randint(low=1, high=vocabulary_size, size=(batch_size, embedding_size))
    describe(tensor)
    output = model(x_in=tensor, apply_softmax=False)
    describe(output)

    output = model(x_in=tensor, apply_softmax=True)
    describe(output)