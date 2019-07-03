from torch.utils.data import Dataset, DataLoader


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


# To use this class you will need to manually install pandas
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
