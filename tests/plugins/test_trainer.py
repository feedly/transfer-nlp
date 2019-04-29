import unittest
from typing import Dict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ignite

from transfer_nlp.common.tokenizers import CharacterTokenizer
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary
from transfer_nlp.plugins.config import register_plugin, ExperimentConfig
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.plugins.regularizers import L1
from transfer_nlp.plugins.trainers import BasicTrainer


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


EXPERIMENT = {
    "data_file": Path(__file__).parent.resolve() / "sample_data.csv",
    "hidden_dim": 100,
    "seed": 1337,
    "lr": 0.001,
    "batch_size": 40,
    "num_epochs": 5,
    "early_stopping_criteria": 5,
    "alpha": 0.01,
    "gradient_clipping": 0.25,
    "mode": "min",
    "factor": 0.5,
    "patience": 1,
    "vectorizer": {
        "_name": "TestVectorizer"
    },
    "dataset_hyper_params": {
        "_name": "TestDatasetHyperParams"
    },
    "dataset_splits": {
        "_name": "TestDataset"
    },
    "model": {
        "_name": "TestModel"
    },
    "model_hyper_params": {
        "_name": "TestHyperParams"
    },
    "model_params": {
        "_name": "TrainableParameters"
    },
    "loss": {
        "_name": "CrossEntropyLoss"
    },
    "optimizer": {
        "_name": "Adam",
        "params": "model_params"
    },
    "regularizer": {
        "_name": "L1"
    },
    "scheduler": {
        "_name": "ReduceLROnPlateau"
    },
    "accuracy": {
        "_name": "Accuracy"
    },
    "lossMetric": {
        "_name": "LossMetric",
        "loss_fn": "loss"
    },
    "trainer": {
        "_name": "BasicTrainer",
        "metrics": [
            "accuracy",
            "lossMetric"
        ]
    },
    "finetune": False
}


class RegistryTest(unittest.TestCase):

    def test_config(self):
        e = ExperimentConfig(EXPERIMENT)
        trainer = e.experiment['trainer']

        self.assertIsInstance(trainer.model, TestModel)
        self.assertIsInstance(trainer.dataset_splits, TestDataset)
        self.assertIsInstance(trainer.loss, torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        self.assertEqual(len(trainer.metrics), 2)
        self.assertEqual(trainer.device, torch.device(type='cpu'))
        self.assertEqual(trainer.seed, 1337)
        self.assertEqual(trainer.loss_accumulation_steps, 4)
        self.assertIsInstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(trainer.num_epochs, EXPERIMENT['num_epochs'])
        self.assertIsInstance(trainer.regularizer, L1)
        self.assertEqual(trainer.gradient_clipping, EXPERIMENT['gradient_clipping'])
        # self.assertEqual(trainer.finetune, False)
        self.assertEqual(trainer.embeddings_name, None)
        self.assertEqual(trainer.forward_params, ['x_in', 'apply_softmax'])
        # trainer.train()

        # Test factories
        optimizer = trainer.experiment_config.factories['optimizer'].create()
        self.assertIsInstance(optimizer, torch.optim.Adam)

        trainer = trainer.experiment_config.factories['trainer'].create()
        self.assertIsInstance(trainer, BasicTrainer)

    def test_setup(self):

        e = ExperimentConfig(EXPERIMENT)
        trainer = e.experiment['trainer']
        trainer.setup(training_metrics=trainer.training_metrics)

        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.EPOCH_COMPLETED]), 5)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.ITERATION_COMPLETED]), 11)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.COMPLETED]), 2)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.STARTED]), 0)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.EPOCH_STARTED]), 5)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.ITERATION_STARTED]), 0)

    def test_forward(self):

        e = ExperimentConfig(EXPERIMENT)
        trainer = e.experiment['trainer']
        trainer.setup(training_metrics=trainer.training_metrics)

        batch = next(iter(trainer.dataset_splits.train_data_loader()))
        self.assertEqual(list(batch.keys()), ['x_in', 'y_target'])
        output = trainer._forward(batch=batch)
        self.assertEqual(output.size()[0], min(len(trainer.dataset_splits.train_set), e.experiment['batch_size']))
        self.assertEqual(output.size()[1], trainer.model.output_dim)
