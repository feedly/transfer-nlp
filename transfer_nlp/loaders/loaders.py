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
