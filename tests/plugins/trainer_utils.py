import numpy as np
import pandas as pd
import torch

from transfer_nlp.common.tokenizers import CharacterTokenizer
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams


@register_plugin
class TestVectorizer(Vectorizer):

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
class TestDatasetHyperParams(ObjectHyperParams):

    def __init__(self, vectorizer: Vectorizer):
        super().__init__()
        self.vectorizer = vectorizer


@register_plugin
class TestDataset(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: TestDatasetHyperParams):
        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: Vectorizer = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
class TestHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: DatasetSplits):
        super().__init__()
        self.input_dim = len(dataset_splits.vectorizer.data_vocab)
        self.output_dim = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class TestModel(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, hidden_dim: int):
        super(TestModel, self).__init__()

        self.input_dim = model_hyper_params.input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = model_hyper_params.output_dim

        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=self.output_dim)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:

        intermediate = torch.nn.functional.relu(self.fc(x_in))
        output = self.fc2(intermediate)

        if self.output_dim == 1:
            output = output.squeeze()

        if apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)

        return output
