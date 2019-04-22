import logging
from pathlib import Path
from typing import Any, Tuple, List, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transfer_nlp.common.tokenizers import CharacterTokenizer
from transfer_nlp.loaders.vocabulary import Vocabulary, SequenceVocabulary
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset, DatasetHyperParams
from transfer_nlp.loaders.vectorizers import SurnamesVectorizer, SurnamesVectorizerCNN, Vectorizer, VectorizerNew
from transfer_nlp.plugins.config import register_plugin, ExperimentConfig
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.predictors.predictor import Predictor, PredictorHyperParams
from transfer_nlp.models.rnn import ElmanRNN, column_gather


# Vectorizers
@register_plugin
class SurnamesVectorizerNew(VectorizerNew):

    def __init__(self, data_file: str):

        super().__init__(data_file=data_file)
        self.tokenizer = CharacterTokenizer()

        df = pd.read_csv(data_file)
        data_vocab = Vocabulary(unk_token='@')
        target_vocab = Vocabulary(add_unk=False)

        # Add surnames and nationalities to vocabulary
        for index, row in df.iterrows():
            surname = row.surname
            nationality = row.nationality
            data_vocab.add_many(tokens=self.tokenizer.tokenize(text=surname))
            target_vocab.add_token(token=nationality)

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab

    def vectorize(self, input_string: str) -> np.array:

        encoding = np.zeros(shape=len(self.data_vocab), dtype=np.float32)
        tokens = self.tokenizer.tokenize(text=input_string)
        for character in tokens:
            encoding[self.data_vocab.lookup_token(token=character)] = 1

        return encoding


@register_plugin
class SurnamesVectorizerCNNNew(VectorizerNew):

    def __init__(self, data_file: str):

        super().__init__(data_file=data_file)

        self.tokenizer = CharacterTokenizer()
        df = pd.read_csv(data_file)
        data_vocab = Vocabulary(unk_token='@')
        target_vocab = Vocabulary(add_unk=False)
        max_surname = 0

        # Add surnames and nationalities to vocabulary
        for index, row in df.iterrows():
            surname = row.surname
            nationality = row.nationality
            data_vocab.add_many(tokens=self.tokenizer.tokenize(text=surname))
            target_vocab.add_token(token=nationality)
            max_surname = max(max_surname, len(surname))
        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self._max_surname = max_surname

    def vectorize(self, input_string: str) -> np.array:

        encoding = np.zeros(shape=(len(self.data_vocab), self._max_surname), dtype=np.float32)
        tokens = self.tokenizer.tokenize(text=input_string)
        for char_index, character in enumerate(tokens):
            encoding[self.data_vocab.lookup_token(token=character)][char_index] = 1

        return encoding


@register_plugin
class SurnameVectorizerRNNNew(VectorizerNew):

    def __init__(self, data_file: str):
        super().__init__(data_file=data_file)
        self.tokenizer = CharacterTokenizer()
        df = pd.read_csv(data_file)

        data_vocab = SequenceVocabulary()
        target_vocab = Vocabulary(add_unk=False)

        max_surname = 0
        for index, row in df.iterrows():
            data_vocab.add_many(tokens=self.tokenizer.tokenize(text=row.surname))
            target_vocab.add_token(row.nationality)
            max_surname = max(max_surname, len(row.surname))

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self._max_surname = max_surname + 2

    def vectorize(self, surname: str) -> Tuple[np.array, int]:
        tokens = self.tokenizer.tokenize(text=surname)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)
        vector_length = self._max_surname

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector, len(indices)


@register_plugin
class SurnameVectorizerGeneration1(VectorizerNew):

    def __init__(self, data_file: str):
        super().__init__(data_file=data_file)
        self.tokenizer = CharacterTokenizer()
        df = pd.read_csv(data_file)

        data_vocab = SequenceVocabulary()
        target_vocab = Vocabulary(add_unk=False)

        max_surname = 0
        for index, row in df.iterrows():
            tokens = self.tokenizer.tokenize(row.surname)
            max_surname = max(max_surname, len(tokens))
            data_vocab.add_many(tokens=tokens)
            target_vocab.add_token(row.nationality)

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self._max_surname = max_surname + 2

    def vectorize(self, surname: str) -> Tuple[np.array, np.array]:
        tokens = self.tokenizer.tokenize(text=surname)

        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)

        vector_length = self._max_surname
        #
        # if vector_length < 0:
        #     vector_length = len(indices)

        from_vector = np.empty(shape=vector_length, dtype=np.int64)
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self.data_vocab.mask_index

        to_vector = np.empty(shape=vector_length, dtype=np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.data_vocab.mask_index

        return from_vector, to_vector


# Dataset classes
@register_plugin
class SurnamesDatasetSplits(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: VectorizerNew = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

        # Class weights
        class_counts = self.df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self.vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict:
        row = self._target_df.iloc[index]

        surname_vector = self.vectorizer.vectorize(input_string=row.surname)

        nationality_index = self.vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index}


@register_plugin
class SurnamesCNNDatasetSplits(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: VectorizerNew = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

        # Class weights
        class_counts = self.df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self.vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict:
        row = self._target_df.iloc[index]

        surname_vector = self.vectorizer.vectorize(input_string=row.surname)

        nationality_index = self.vectorizer.target_vocab.lookup_token(row.nationality)
        print(f"row: {row}")
        print(f"surname vector: {surname_vector}")
        print(f"nationality index: {nationality_index}")

        return {
            'x_in': surname_vector,
            'y_target': nationality_index}


@register_plugin
class SurnamesRNNDatasetSplits(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: VectorizerNew = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['x_lengths'] = self.df.apply(lambda row: row.x_in[1], axis=1)
        self.df['x_in'] = self.df.apply(lambda row: row.x_in[0], axis=1)

        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target', 'x_lengths']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target', 'x_lengths']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target', 'x_lengths']]

        self.tokenizer = CharacterTokenizer()

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

        # Class weights
        class_counts = self.df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self.vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname_vector, vec_length = self.vectorizer.vectorize(surname=row.surname)
        nationality_index = self.vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index,
            'x_lengths': vec_length}


@register_plugin
class SurnameHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: SurnamesDatasetSplits):
        super().__init__()
        self.input_dim = len(dataset_splits.vectorizer.data_vocab)
        self.output_dim = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class SurnameDatasetGeneration(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: VectorizerNew = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: row.x_in[1], axis=1)
        self.df['x_in'] = self.df.apply(lambda row: row.x_in[0], axis=1)

        self.df['nationality_index'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target', 'nationality_index']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target', 'nationality_index']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target', 'nationality_index']]

        self.tokenizer = CharacterTokenizer()

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        x_in, y_target = self.vectorizer.vectorize(surname=row.surname)
        nationality_index = self.vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': x_in,
            'y_target': y_target,
            'nationality_index': nationality_index}


# Models
@register_plugin
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, hidden_dim: int):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = model_hyper_params.input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = model_hyper_params.output_dim

        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=self.output_dim)
        # TODO: experiment with more layers

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        """
        Linear -> ReLu -> Linear (+ softmax if probabilities needed)
        :param x_in: size (batch, input_dim)
        :param apply_softmax: False if used with the cross entropy loss, True if probability wanted
        :return:
        """
        # TODO: experiment with other activation functions

        intermediate = torch.nn.functional.relu(self.fc(x_in))
        output = self.fc2(intermediate)

        if self.output_dim == 1:
            output = output.squeeze()

        if apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)

        return output


@register_plugin
class SurnameCNNHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: SurnamesDatasetSplits):
        super().__init__()
        self.initial_num_channels = len(dataset_splits.vectorizer.data_vocab)
        self.num_classes = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class SurnameClassifierCNN1(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, num_channels: int):
        super(SurnameClassifierCNN1, self).__init__()

        self.initial_num_channels: int = model_hyper_params.initial_num_channels
        self.num_classes: int = model_hyper_params.num_classes
        self.num_channels: int = num_channels

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.initial_num_channels,
                            out_channels=self.num_channels, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3),
            torch.nn.ELU()
        )
        self.fc = torch.nn.Linear(self.num_channels, self.num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        """
        Conv -> ELU -> ELU -> Conv -> ELU -> Linear
        :param x_in: size (batch, initial_num_channels, max_sequence)
        :param apply_softmax: False if used with the cross entropy loss, True if probability wanted
        :return:
        """
        features = self.convnet(x_in).squeeze(dim=2)

        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = torch.nn.functional.softmax(prediction_vector, dim=1)

        return prediction_vector


@register_plugin
class SurnameRNNHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: SurnamesRNNDatasetSplits):
        super().__init__()
        self.num_embeddings = len(dataset_splits.vectorizer.data_vocab)
        self.num_classes = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class SurnameClassifierRNN2(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, embedding_size: int,
                 rnn_hidden_size: int, batch_first: bool = True, padding_idx: int = 0):

        super(SurnameClassifierRNN2, self).__init__()
        self.num_embeddings = model_hyper_params.num_embeddings
        self.num_classes = model_hyper_params.num_classes

        self.emb: torch.nn.Embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                                    embedding_dim=embedding_size,
                                                    padding_idx=padding_idx)
        self.rnn: ElmanRNN = ElmanRNN(input_size=embedding_size,
                                      hidden_size=rnn_hidden_size,
                                      batch_first=batch_first)
        self.fc1: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size,
                                                    out_features=rnn_hidden_size)
        self.fc2: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size,
                                                    out_features=self.num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x_in: torch.Tensor, x_lengths: torch.Tensor = None, apply_softmax: bool = False) -> torch.Tensor:
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

        y_out = torch.nn.functional.relu(self.fc1(self.dropout(y_out)))
        y_out = self.fc2(self.dropout(y_out))

        if apply_softmax:
            y_out = torch.nn.functional.softmax(y_out, dim=1)

        return y_out


@register_plugin
class SurnameGenerationHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: SurnamesRNNDatasetSplits):
        super().__init__()
        self.char_vocab_size = len(dataset_splits.vectorizer.data_vocab)
        self.num_nationalities = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class SurnameConditionedGenerationModel(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, char_embedding_size: int, rnn_hidden_size: int, batch_first: bool = True, padding_idx: int = 0,
                 dropout_p: float = 0.5,
                 conditioned: bool = False):

        super(SurnameConditionedGenerationModel, self).__init__()
        self.char_vocab_size = model_hyper_params.char_vocab_size
        self.num_nationalities = model_hyper_params.num_nationalities

        self.char_emb: torch.nn.Embedding = torch.nn.Embedding(num_embeddings=self.char_vocab_size,
                                                               embedding_dim=char_embedding_size,
                                                               padding_idx=padding_idx)
        self.nation_emb: torch.nn.Embedding = None
        self.conditioned = conditioned
        if self.conditioned:
            self.nation_emb = torch.nn.Embedding(num_embeddings=self.num_nationalities,
                                                 embedding_dim=rnn_hidden_size)
        self.rnn: torch.nn.GRU = torch.nn.GRU(input_size=char_embedding_size,
                                              hidden_size=rnn_hidden_size,
                                              batch_first=batch_first)
        self.fc: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size,
                                                   out_features=self.char_vocab_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x_in: torch.Tensor, nationality_index: int = 0, apply_softmax: bool = False) -> torch.Tensor:
        """

        :param x_in: input data tensor, x_in.shape should be (batch, max_seq_size)
        :param nationality_index: The index of the nationality for each data point
                Used to initialize the hidden state of the RNN
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, char_vocab_size)
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
        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = torch.nn.functional.softmax(y_out, dim=1)

        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)

        return y_out


# Predictors
@register_plugin
class MLPPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, predictor_hyper_params: PredictorHyperParams):
        super().__init__(predictor_hyper_params=predictor_hyper_params)

    def json_to_data(self, input_json: Dict):
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


@register_plugin
class SurnameCNNPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, predictor_hyper_params: PredictorHyperParams):
        super().__init__(predictor_hyper_params=predictor_hyper_params)

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


@register_plugin
class SurnameRNNPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, predictor_hyper_params: PredictorHyperParams):
        super().__init__(predictor_hyper_params=predictor_hyper_params)

    def json_to_data(self, input_json: Dict) -> Dict:
        # vector_length = 30

        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(surname=input_string)[0] for input_string in input_json['inputs']]),
            'x_lengths': torch.Tensor([self.vectorizer.vectorize(surname=input_string)[1] for input_string in input_json['inputs']])
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
    logging.basicConfig(level=logging.INFO)

    import argparse

    experiment = ExperimentConfig.from_json('mlp.json', HOME=str(Path.home()))
    experiment['trainer'].train()
    #
    # if slack_webhook_url and slack_webhook_url != "YourWebhookURL":
    #     run_with_slack(runner=runner, test_at_the_end=True)
    # else:
    #     # runner.run(test_at_the_end=True)
    #     runner.run_pipeline()
