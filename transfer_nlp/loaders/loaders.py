"""
This file contains an abstract CustomDataset class, on which we can build up custom dataset classes.

We include 2 dataset classes:

- ReviewsDataset --> Yelp reviews dataset
- SurnamesDataset --> Surnames dataset
"""

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from loaders.vectorizers import ReviewsVectorizer, SurnamesVectorizer, Vectorizer, SurnamesVectorizerCNN, CBOWVectorizer, NewsVectorizer, SurnameVectorizerRNN, \
    SurnameVectorizerGeneration, NMTVectorizer, FeedlyVectorizer
from common.tokenizers import tokenize


class CustomDataset(Dataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        self.dataset_df: pd.DataFrame = dataset_df
        self._vectorizer: Vectorizer = vectorizer
        self.train_df = self.dataset_df[self.dataset_df.split == 'train']
        self.train_size = len(self.train_df)
        self.val_df = self.dataset_df[self.dataset_df.split == 'val']
        self.val_size = len(self.val_df)
        self.test_df = self.dataset_df[self.dataset_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict: Dict = {'train': (self.train_df, self.train_size),
                                   'val': (self.val_df, self.val_size),
                                   'test': (self.test_df, self.test_size)}
        self._target_split: str = ''
        self._target_df: pd.DataFrame = None
        self._target_size: int = 0

        self.set_split(split='train')

    def save_vectorizer(self, vectorizer_filepath: Path):

        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self) -> Vectorizer:

        return self._vectorizer

    def set_split(self, split: str = 'train'):

        self._target_split: str = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def get_num_batches(self, batch_size: int) -> float:

        return len(self) // batch_size

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_df, vectorizer_filepath) -> 'CustomDataset':
        dataset_df = pd.read_csv(dataset_df)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath=vectorizer_filepath)
        return cls(dataset_df=dataset_df, vectorizer=vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: str) -> 'CustomDataset':
        pass

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:
        pass

    def __len__(self):
        return self._target_size


class ReviewsDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: str) -> CustomDataset:

        dataset_df = pd.read_csv(filepath_or_buffer=dataset_csv)
        train_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=ReviewsVectorizer.from_dataframe(review_df=train_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:

        with open(vectorizer_filepath) as fp:
            return ReviewsVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(input_string=row.review)

        rating_index = self._vectorizer.target_vocab.lookup_token(row.rating)

        return {'x_in': review_vector,
                'y_target': rating_index}


class SurnamesDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        # Class weights
        class_counts = dataset_df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self._vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: Path) -> CustomDataset:
        dataset_df = pd.read_csv(filepath_or_buffer=dataset_csv)
        train_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=SurnamesVectorizer.from_dataframe(train_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:

        with open(vectorizer_filepath) as fp:
            return SurnamesVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]

        surname_vector = self._vectorizer.vectorize(input_string=row.surname)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index}


class SurnamesDatasetCNN(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        # Class weights
        class_counts = dataset_df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self._vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: Path) -> CustomDataset:
        dataset_df = pd.read_csv(filepath_or_buffer=dataset_csv)
        train_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=SurnamesVectorizerCNN.from_dataframe(train_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:

        with open(vectorizer_filepath) as fp:
            return SurnamesVectorizerCNN.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]

        surname_vector = self._vectorizer.vectorize(input_string=row.surname)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index}


class CBOWDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, dataset_df.context))

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: Path) -> CustomDataset:

        dataset_df = pd.read_csv(filepath_or_buffer=dataset_csv)
        train_cbow_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df, CBOWVectorizer.from_dataframe(train_cbow_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:

        with open(vectorizer_filepath) as fp:
            return CBOWVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]

        context_vector = self._vectorizer.vectorize(row.context, self._max_seq_length)
        target_index = self._vectorizer.data_vocab.lookup_token(row.target)

        return {
            'x_in': context_vector,
            'y_target': target_index}


class NewsDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):
        """
        Args:
            news_df (pandas.DataFrame): the dataset
            vectorizer (NewsVectorizer): vectorizer instatiated from dataset
        """
        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, dataset_df.title)) + 2

        # Class weights
        class_counts = dataset_df.category.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self._vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv: Path):
        dataset_df = pd.read_csv(news_csv)
        train_news_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=NewsVectorizer.from_dataframe(train_news_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path):

        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index):

        row = self._target_df.iloc[index]
        title_vector = self._vectorizer.vectorize(title=row.title, vector_length=self._max_seq_length)

        category_index = self._vectorizer.target_vocab.lookup_token(row.category)

        return {
            'x_in': title_vector,
            'y_target': category_index}


class SurnameDatasetRNN(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        self._max_seq_length = max(map(len, self.dataset_df.surname)) + 2
        class_counts = self.train_df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self._vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv: Path):
        """Load dataset and make a new vectorizer from scratch

        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        dataset_df = pd.read_csv(surname_csv)
        train_surname_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=SurnameVectorizerRNN.from_dataframe(train_surname_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path):

        with open(vectorizer_filepath) as fp:
            return SurnameVectorizerRNN.from_serializable(json.load(fp))

    def __getitem__(self, index):

        row = self._target_df.iloc[index]
        surname_vector, vec_length = self._vectorizer.vectorize(surname=row.surname, vector_length=self._max_seq_length)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index,
            'x_lengths': vec_length}


class SurnameDatasetGeneration(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        self._max_seq_length = max(map(len, self.dataset_df.surname)) + 2

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv: Path):

        dataset_df = pd.read_csv(surname_csv)
        train_surname_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=SurnameVectorizerGeneration.from_dataframe(train_surname_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path):

        with open(vectorizer_filepath) as fp:
            return SurnameVectorizerGeneration.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self._target_df.iloc[index]

        from_vector, to_vector = self._vectorizer.vectorize(surname=row.surname, vector_length=self._max_seq_length)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': from_vector,
            'y_target': to_vector,
            'nationality_index': nationality_index}


class FeedlyDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):

        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

        self._max_seq_length = 0# = max(map(len, self.dataset_df.content)) + 2
        for index, row in dataset_df.iterrows():
            content = [token for token in tokenize(row.content)]
            self._max_seq_length = max(len(content), self._max_seq_length)
        self._max_seq_length += 2

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv: Path):

        dataset_df = pd.read_csv(surname_csv)
        train_surname_df = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=FeedlyVectorizer.from_dataframe(train_surname_df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path):

        with open(vectorizer_filepath) as fp:
            return FeedlyVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self._target_df.iloc[index]

        from_vector, to_vector = self._vectorizer.vectorize(content=row.content, vector_length=self._max_seq_length)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_data': from_vector,
            'y_target': to_vector,
            'class_index': nationality_index}

class NMTDataset(CustomDataset):

    def __init__(self, dataset_df: pd.DataFrame, vectorizer: NMTVectorizer):
        super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv: Path):

        dataset_df = pd.read_csv(filepath_or_buffer=dataset_csv)
        train_subset = dataset_df[dataset_df.split == 'train']
        return cls(dataset_df=dataset_df, vectorizer=NMTVectorizer.from_dataframe(train_subset))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: Path):

        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self._target_df.iloc[index]
        vector_dict = self._vectorizer.vectorize(source_text=row.source_language, target_text=row.target_language)

        return {
            "x_source": vector_dict["source_vector"],
            "target_sequence": vector_dict["target_x_vector"],
            "y_target": vector_dict["target_y_vector"],
            "x_source_lengths": vector_dict["source_length"]}


def generate_batches(dataset: Dataset, batch_size: int, shuffle: bool=True, drop_last: bool=True, device: str='cpu') -> Dict:

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def generate_nmt_batches(dataset: NMTDataset, batch_size: int, shuffle: bool=True,
                         drop_last: bool=True, device: str="cpu") -> Dict:
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_lengths'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict