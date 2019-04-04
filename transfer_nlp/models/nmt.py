from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate import bleu_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transfer_nlp.common.utils import describe
from transfer_nlp.loaders.vectorizers import NMTVectorizer
from transfer_nlp.loaders.vocabulary import SequenceVocabulary

chencherry = bleu_score.SmoothingFunction()


class NMTEncoder(nn.Module):

    def __init__(self, num_embeddings: int, embedding_size: int, rnn_hidden_size: int):

        super(NMTEncoder, self).__init__()

        self.source_embedding: nn.Embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.birnn: nn.GRU = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_source: torch.Tensor, x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the model

        Args:
            x_source (torch.Tensor): the input data tensor.
                x_source.shape is (batch, seq_size)
            x_lengths (torch.Tensor): a vector of lengths for each item in the batch
        Returns:
            a tuple: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h.shape = (batch, rnn_hidden_size * 2)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(),
                                        batch_first=True)

        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out, x_birnn_h = self.birnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        #  (recall: -1 takes the remaining positions,
        #           flattening the two RNN hidden vectors into 1)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h


def verbose_attention(encoder_state_vectors: torch.Tensor, query_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size),
                              dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores


def terse_attention(encoder_state_vectors: torch.Tensor, query_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1),
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities


class NMTDecoder(nn.Module):

    def __init__(self, num_embeddings: int, embedding_size: int, rnn_hidden_size: int, bos_index: int):

        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size: int = rnn_hidden_size
        self.target_embedding: nn.Embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=0)
        self.gru_cell: nn.GRUCell = nn.GRUCell(input_size=embedding_size + rnn_hidden_size,
                                   hidden_size=rnn_hidden_size)
        self.hidden_map: nn.Linear = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.classifier: nn.Linear = nn.Linear(in_features=rnn_hidden_size * 2, out_features=num_embeddings)
        self.bos_index: int = bos_index
        self._sampling_temperature: int = 3

    def _init_indices(self, batch_size: int) -> torch.Tensor:
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index

    def _init_context_vectors(self, batch_size: int) -> torch.Tensor:
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def forward(self, encoder_state: torch.Tensor, initial_hidden_state: torch.Tensor, target_sequence: torch.Tensor, sample_probability: float=0.0) -> torch.Tensor:


        if target_sequence is None:
            sample_probability = 1.0
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)


        # target_sequence = target_sequence.permute(1, 0)
        # output_sequence_size = target_sequence.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size=batch_size)
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            # Schedule sampling is whe
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            # y_t_index = target_sequence[i]

            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)  #TODO: change List to Tuple

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())

            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state,
                                                           query_vector=h_t)

            # auxillary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            # Step 4: Use the current hidden and context vectors to make a prediction to the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                # _, y_t_index = torch.max(p_y_t_index, 1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            # auxillary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors


class NMTModel(nn.Module):
    """ The Neural Machine Translation Model """

    def __init__(self, source_vocab_size: int, source_embedding_size: int,
                 target_vocab_size: int, target_embedding_size: int, encoding_size: int,
                 target_bos_index: int):

        super(NMTModel, self).__init__()
        self.encoder: NMTEncoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder: NMTDecoder = NMTDecoder(num_embeddings=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index)

    def forward(self, x_source: torch.Tensor, x_source_lengths: torch.Tensor, target_sequence: torch.Tensor, sample_probability: float=0.0) -> torch.Tensor:

        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state,
                                      initial_hidden_state=final_hidden_states,
                                      target_sequence=target_sequence,
                                      sample_probability=sample_probability)
        return decoded_states


def sentence_from_indices(indices: List[int], vocab: SequenceVocabulary, strict: bool=True, return_string: bool=True):

    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        elif index not in ignore_indices:
            out.append(vocab.lookup_index(index))
        else:
            pass

    if return_string:
        return " ".join(out)
    else:
        return out


class NMTSampler:

    def __init__(self, vectorizer: NMTVectorizer, model: NMTModel):

        self.vectorizer: NMTVectorizer = vectorizer
        self.model: NMTModel = model

    def apply_to_batch(self, batch_dict: Dict[str, Any]):

        self._last_batch = batch_dict
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_length'],
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred

        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched

    def _get_source_sentence(self, index: int, return_string: bool=True):

        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.data_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):

        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_sampled_sentence(self, index: int, return_string: bool=True):

        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index: int, return_string: bool=True) -> Dict[str, Any]:

        output = {
            "source": self._get_source_sentence(index=index, return_string=return_string),
            "reference": self._get_reference_sentence(index=index, return_string=return_string),
            "sampled": self._get_sampled_sentence(index=index, return_string=return_string),
            "attention": self._last_batch['attention'][index]}

        reference = output['reference']
        hypothesis = output['sampled']

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)

        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)

        return output


if __name__ == "__main__":

    batch_size = 32
    num_embeddings = 100
    embedding_size = 100
    rnn_hidden_size = 100
    sequence_size = 100

    model = NMTEncoder(num_embeddings=num_embeddings, embedding_size=embedding_size, rnn_hidden_size=rnn_hidden_size)

    tensor = torch.randint(low=1, high=num_embeddings, size=(batch_size, sequence_size))
    describe(tensor)
    lens = torch.randint(low=1, high=num_embeddings, size=(batch_size,))
    lens = torch.sort(input=lens, descending=True)[0]
    describe(lens)
    x_unpacked, x_birnn_h = model(x_source=tensor, x_lengths=lens)
    describe(x=x_unpacked)
    describe(x=x_birnn_h)
