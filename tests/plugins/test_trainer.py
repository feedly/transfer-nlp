import unittest
from pathlib import Path

import ignite

from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.plugins.regularizers import L1
from transfer_nlp.plugins.trainers import BasicTrainer
from .trainer_utils import *

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
