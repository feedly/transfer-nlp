import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders.vectorizers import Vectorizer
from common.utils import describe

name = 'transfer_nlp.models.generation'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class SurnameConditionedGenerationModel(nn.Module):

    def __init__(self, char_embedding_size: int, char_vocab_size: int, num_nationalities: int,
                 rnn_hidden_size: int, batch_first: bool=True, padding_idx: int=0, dropout_p: float=0.5, conditioned: bool=False):

        super(SurnameConditionedGenerationModel, self).__init__()

        self.char_emb: nn.Embedding = nn.Embedding(num_embeddings=char_vocab_size,
                                     embedding_dim=char_embedding_size,
                                     padding_idx=padding_idx)
        self.nation_emb: nn.Embedding = None
        self.conditioned = conditioned
        if self.conditioned:
            self.nation_emb = nn.Embedding(num_embeddings=num_nationalities,
                                       embedding_dim=rnn_hidden_size)
        self.rnn: nn.GRU = nn.GRU(input_size=char_embedding_size,
                          hidden_size=rnn_hidden_size,
                          batch_first=batch_first)
        self.fc: nn.Linear = nn.Linear(in_features=rnn_hidden_size,
                            out_features=char_vocab_size)
        self._dropout_p: float = dropout_p

    def forward(self, x_in: torch.Tensor, nationality_index: int=0, apply_softmax: bool=False) -> torch.Tensor:
        """The forward pass of the model

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, max_seq_size)
            nationality_index (torch.Tensor): The index of the nationality for each data point
                Used to initialize the hidden state of the RNN
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, char_vocab_size)
        """
        x_embedded = self.char_emb(x_in)
        # hidden_size: (num_layers * num_directions, batch_size, rnn_hidden_size)
        if self.conditioned:
            nationality_embedded = self.nation_emb(nationality_index).unsqueeze(0)
            y_out, _ = self.rnn(x_embedded, nationality_embedded)
        else:
            y_out, _ = self.rnn(x_embedded)

        batch_size, seq_size, feat_size = y_out.shape
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        y_out = self.fc(F.dropout(y_out, p=self._dropout_p))

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)

        return y_out


def sample_from_model(model: SurnameConditionedGenerationModel, vectorizer: Vectorizer, num_samples: int=1, sample_size: int=20,
                      temperature: float=1.0) -> torch.Tensor:

    begin_seq_index = [vectorizer.data_vocab.begin_seq_index
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index,
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    h_t = None

    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.char_emb(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)

    return indices


def sample_from_conditioned_model(model: SurnameConditionedGenerationModel, vectorizer: Vectorizer, nationalities: List[int], sample_size: int=20,
                      temperature: float=1.0) -> torch.Tensor:

    num_samples = len(nationalities)
    begin_seq_index = [vectorizer.data_vocab.begin_seq_index
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index,
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    nationality_indices = torch.tensor(nationalities, dtype=torch.int64).unsqueeze(dim=0)
    h_t = model.nation_emb(nationality_indices)

    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.char_emb(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices


def decode_samples(sampled_indices: torch.Tensor, vectorizer: Vectorizer, character: bool=True) -> List[str]:

    decoded_surnames = []
    vocab = vectorizer.data_vocab

    for sample_index in range(sampled_indices.shape[0]):
        surname = []
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index, time_step].item()
            if sample_item == vocab.begin_seq_index:
                continue
            elif sample_item == vocab.end_seq_index:
                break
            else:
                # print(vocab.lookup_index(sample_item))
                surname += [vocab.lookup_index(sample_item)]

        if character:
            surname = ''.join(surname)
        else:
            surname = ' '.join(surname)
        decoded_surnames.append(surname)

    return decoded_surnames


def generate_names(model: SurnameConditionedGenerationModel, vectorizer: Vectorizer, character: bool=True):

    model = model.cpu()

    if model.conditioned:
        logger.info("Generating surnames conditioned on the nationality")
        for index in range(len(vectorizer.target_vocab)):
            nationality = vectorizer.target_vocab.lookup_index(index)
            logger.info("Sampled for {}: ".format(nationality))
            sampled_indices = sample_from_conditioned_model(model=model, vectorizer=vectorizer,
                                                            nationalities=[index] * 3,
                                                            temperature=0.7)
            for sampled_surname in decode_samples(sampled_indices=sampled_indices, vectorizer=vectorizer, character=character):
                logger.info("-  " + sampled_surname)

    else:
        logger.info("Generating surnames unconditioned on the nationality")
        num_names = 10
        # Generate nationality hidden state
        sampled_surnames = decode_samples(
            sample_from_model(model=model, vectorizer=vectorizer, num_samples=num_names, sample_size=20, temperature=1.0),
            vectorizer=vectorizer, character=character)
        # Show results
        logger.info("-" * 15)
        for i in range(num_names):
            logger.info(sampled_surnames[i])


if __name__ == "__main__":

    char_embedding_size = 32
    char_vocab_size = 256
    rnn_hidden_size = 100
    num_nationalities = 2
    batch_size = 32
    max_sequence = 100

    model = SurnameConditionedGenerationModel(char_embedding_size=char_embedding_size, char_vocab_size=char_vocab_size, rnn_hidden_size=rnn_hidden_size, num_nationalities=num_nationalities, conditioned=True)
    print(model)
    tensor = torch.ones(size=(batch_size, max_sequence)).long()
    describe(tensor)
    nationality_index = torch.zeros(size=(batch_size,), dtype=torch.int64)
    output = model(x_in=tensor, nationality_index=nationality_index)
    describe(output)
