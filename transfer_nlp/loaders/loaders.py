"""
This file contains an abstract CustomDataset class, on which we can build up custom dataset classes.

In your project, you will have to customize your data loader class. To let the framework interact with your class, you
need to use the decorator @register_dataset, just as in the examples in this file
"""

from torch.utils.data import Dataset, DataLoader

from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams


@register_plugin
class DatasetHyperParams(ObjectHyperParams):

    def __init__(self, vectorizer: Vectorizer):
        super().__init__()
        self.vectorizer = vectorizer


class DatasetSplits:
    def __init__(self,
                 train_set: Dataset, train_batch_size: int,
                 val_set: Dataset, val_batch_size: int,
                 test_set: Dataset = None, test_batch_size: int = None):
        self.train_set: Dataset = train_set
        self.train_batch_size: int = train_batch_size

        self.val_set: Dataset = val_set
        self.val_batch_size: int = val_batch_size

        self.test_set: Dataset = test_set
        self.test_batch_size: int = test_batch_size

    def train_data_loader(self):
        return DataLoader(self.train_set, self.train_batch_size, shuffle=True)

    def val_data_loader(self):
        return DataLoader(self.val_set, self.val_batch_size, shuffle=False)

    def test_data_loader(self):
        return DataLoader(self.test_set, self.test_batch_size, shuffle=False)


class DataFrameDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item, :]
        return {col: row[col] for col in self.df.columns}


class DataProps:
    def __init__(self):
        self.input_dims: int = None
        self.output_dims: int = None

# TODO: move these to independant experiment python files
# @register_dataset
# class FeedlyDataset(CustomDataset):
#
#     def __init__(self, dataset_df: pd.DataFrame, vectorizer: Vectorizer):
#
#         super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)
#
#         self.tokenizer = CharacterTokenizer()
#
#         measure_len = lambda content: len(self.tokenizer.tokenize(text=content))
#         self._max_seq_length = max(map(measure_len, dataset_df.content)) + 2
#
#     @classmethod
#     def load_dataset_and_make_vectorizer(cls, surname_csv: Path):
#
#         dataset_df = pd.read_csv(surname_csv)
#         train_surname_df = dataset_df[dataset_df.split == 'train']
#         return cls(dataset_df=dataset_df, vectorizer=FeedlyVectorizer.from_dataframe(train_surname_df))
#
#     @staticmethod
#     def load_vectorizer_only(vectorizer_filepath: Path):
#
#         with open(vectorizer_filepath) as fp:
#             return FeedlyVectorizer.from_serializable(json.load(fp))
#
#     def __getitem__(self, index: int) -> Dict[str, Any]:
#
#         row = self._target_df.iloc[index]
#
#         from_vector, to_vector = self._vectorizer.vectorize(content=row.content, vector_length=self._max_seq_length)
#
#         nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)
#
#         return {
#             'x_in': from_vector,
#             'y_target': to_vector,
#             'class_index': nationality_index}
#
#
# @register_dataset
# class NMTDataset(CustomDataset):
#
#     def __init__(self, dataset_df: pd.DataFrame, vectorizer: NMTVectorizer):
#         super().__init__(dataset_df=dataset_df, vectorizer=vectorizer)
#
#     @classmethod
#     def load_dataset_and_make_vectorizer(cls, dataset: Path):
#
#         dataset_df = pd.read_csv(filepath_or_buffer=dataset)
#         train_subset = dataset_df[dataset_df.split == 'train']
#         return cls(dataset_df=dataset_df, vectorizer=NMTVectorizer.from_dataframe(train_subset))
#
#     @staticmethod
#     def load_vectorizer_only(vectorizer_filepath: Path):
#
#         with open(vectorizer_filepath) as fp:
#             return NMTVectorizer.from_serializable(json.load(fp))
#
#     def __getitem__(self, index: int) -> Dict[str, Any]:
#
#         row = self._target_df.iloc[index]
#         vector_dict = self._vectorizer.vectorize(source_text=row.source_language, target_text=row.target_language)
#
#         return {
#             "x_source": vector_dict["source_vector"],
#             "target_sequence": vector_dict["target_x_vector"],
#             "y_target": vector_dict["target_y_vector"],
#             "x_source_lengths": vector_dict["source_length"]}
