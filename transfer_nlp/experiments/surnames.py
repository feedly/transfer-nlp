import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import SurnamesVectorizer
from transfer_nlp.plugins.config import register_plugin, ExperimentConfig
from transfer_nlp.plugins.helpers import ModelHyperParams

@register_plugin
class SurnamesDatasetSplits(DatasetSplits):

    def __init__(self, data_file:str, batch_size: int):

        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer:SurnamesVectorizer = SurnamesVectorizer.from_dataframe(self.df)

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in','y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in','y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in','y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

        # Class weights
        class_counts = self.df.nationality.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self.vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    # @classmethod
    # def load_dataset_and_make_vectorizer(cls, dataset: Path) -> CustomDataset:
    #
    #     dataset_df = pd.read_csv(filepath_or_buffer=dataset)
    #     train_df = dataset_df[dataset_df.split == 'train']
    #
    #     return cls(dataset_df=dataset_df, vectorizer=SurnamesVectorizer.from_dataframe(train_df))
    #
    # @staticmethod
    # def load_vectorizer_only(vectorizer_filepath: Path) -> Vectorizer:
    #
    #     with open(vectorizer_filepath) as fp:
    #         return SurnamesVectorizer.from_serializable(json.load(fp))

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]

        surname_vector = self._vectorizer.vectorize(input_string=row.surname)

        nationality_index = self._vectorizer.target_vocab.lookup_token(row.nationality)

        return {
            'x_in': surname_vector,
            'y_target': nationality_index}

@register_plugin
class SurnameHyperParams(ModelHyperParams):

    def __init__(self, dataset_splits:SurnamesDatasetSplits):
        super().__init__()
        self.input_dim = len(dataset_splits.vectorizer.data_vocab)
        self.output_dim = len(dataset_splits.vectorizer.target_vocab)

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


