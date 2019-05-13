import logging
import string
from collections import Counter
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

from transfer_nlp.common.tokenizers import CustomTokenizer
from transfer_nlp.embeddings.embeddings import Embedding
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary, SequenceVocabulary
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.predictors import PredictorABC

logger = logging.getLogger(__name__)


# Vectorizer class
@register_plugin
class NewsVectorizer(Vectorizer):
    def __init__(self, data_file: str, cutoff: int):

        super().__init__(data_file=data_file)
        self.cutoff = cutoff

        self.tokenizer = CustomTokenizer()
        df = pd.read_csv(data_file)

        target_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(df.category)):
            target_vocab.add_token(category)

        word_counts = Counter()
        max_title = 0
        for title in df.title:
            tokens = self.tokenizer.tokenize(text=title)
            max_title = max(max_title, len(tokens))
            for token in tokens:
                if token not in string.punctuation:
                    word_counts[token] += 1

        data_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= self.cutoff:
                data_vocab.add_token(word)

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self.max_title = max_title + 2

    def vectorize(self, title: str) -> np.array:

        tokens = self.tokenizer.tokenize(text=title)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)
        vector_length = self.max_title

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector


# Dataset class
@register_plugin
class NewsDataset(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, vectorizer: Vectorizer):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: Vectorizer = vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.title), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.category), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
class NewsClassifier(torch.nn.Module):

    def __init__(self, data: DatasetSplits, embedding_size: int, num_channels: int,
                 hidden_dim: int, dropout_p: float, padding_idx: int = 0, glove_path: str = None):
        super(NewsClassifier, self).__init__()

        self.num_embeddings = len(data.vectorizer.data_vocab)
        self.num_classes = len(data.vectorizer.target_vocab)

        self.num_channels: int = num_channels
        self.embedding_size: int = embedding_size
        self.hidden_dim: int = hidden_dim
        self.padding_idx: int = padding_idx

        if glove_path:
            logger.info("Using pre-trained word embeddings...")
            self.embeddings = Embedding(glove_filepath=glove_path, data=data).embeddings
            self.embeddings = torch.from_numpy(self.embeddings).float()
            glove_size = len(self.embeddings[0])
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=glove_size,
                                                              num_embeddings=self.num_embeddings,
                                                              padding_idx=self.padding_idx,
                                                              _weight=self.embeddings)

        else:
            logger.info("Not using pre-trained word embeddings...")
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=self.embedding_size,
                                                              num_embeddings=self.num_embeddings,
                                                              padding_idx=self.padding_idx)

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.embedding_size,
                            out_channels=self.num_channels, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=1),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3),  # Experimental change from 3 to 2
            torch.nn.ELU()
        )

        self._dropout_p: float = dropout_p
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.num_channels, self.hidden_dim)
        self.fc2: torch.nn.Linear = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        """

        :param x_in: input data tensor
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)

        features = self.convnet(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = torch.nn.functional.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = self.dropout(features)

        # mlp classifier
        intermediate_vector = torch.nn.functional.relu(self.dropout(self.fc1(features)))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = torch.nn.functional.softmax(prediction_vector, dim=1)

        return prediction_vector


# Predictors
@register_plugin
class NewsPredictor(PredictorABC):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, data: DatasetSplits, model: torch.nn.Module):
        super().__init__(vectorizer=data.vectorizer, model=model)

    def json_to_data(self, input_json: Dict) -> Dict:
        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(title=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)

        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]
