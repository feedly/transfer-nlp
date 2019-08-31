import copy
import unittest
from pathlib import Path

import ignite
import ignite.metrics as metrics
import torch.nn as nn
import torch.optim as optim
from ignite.metrics import Precision, Recall, MetricsLambda

from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.plugins.regularizers import L1
from .trainer_utils import *

from transfer_nlp.plugins import regularizers, helpers, trainers, metrics as m

PLUGINS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "AdaDelta": optim.Adadelta,
    "AdaGrad": optim.Adagrad,
    "SparseAdam": optim.SparseAdam,
    "AdaMax": optim.Adamax,
    "ASGD": optim.ASGD,
    "LBFGS": optim.LBFGS,
    "RMSPROP": optim.RMSprop,
    "Rprop": optim.Rprop,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR,
    "ReLU": nn.functional.relu,
    "LeakyReLU": nn.functional.leaky_relu,
    "Tanh": nn.functional.tanh,
    "Softsign": nn.functional.softsign,
    "Softshrink": nn.functional.softshrink,
    "Softplus": nn.functional.softplus,
    "Sigmoid": nn.Sigmoid,
    "CELU": nn.CELU,
    "SELU": nn.functional.selu,
    "RReLU": nn.functional.rrelu,
    "ReLU6": nn.functional.relu6,
    "PReLU": nn.functional.prelu,
    "LogSigmoid": nn.functional.logsigmoid,
    "Hardtanh": nn.functional.hardtanh,
    "Hardshrink": nn.functional.hardshrink,
    "ELU": nn.functional.elu,
    "Softmin": nn.functional.softmin,
    "Softmax": nn.functional.softmax,
    "LogSoftmax": nn.functional.log_softmax,
    "GLU": nn.functional.glu,
    "TanhShrink": nn.functional.tanhshrink,
    "Accuracy": metrics.Accuracy,
}
for plugin_name, plugin in PLUGINS.items():
    register_plugin(registrable=plugin, alias=plugin_name)


def fbeta(r, p, beta, average):
    if average:
        return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)
    else:
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()


@register_plugin
def create_fbeta():
    return MetricsLambda(fbeta, Recall(average=True), Precision(average=True), 0.5, True)


EXPERIMENT = {
    "my_dataset_splits": {
        "_name": "TestDataset",
        "data_file": Path(__file__).parent.resolve() / "sample_data.csv",
        "batch_size": 128,
        "vectorizer": {
            "_name": "TestVectorizer",
            "data_file": Path(__file__).parent.resolve() / "sample_data.csv"
        }
    },
    "model": {
        "_name": "TestModel",
        "hidden_dim": 100,
        "data": "$my_dataset_splits"
    },
    "optimizer": {
        "_name": "Adam",
        "lr": 0.01,
        "params": {
            "_name": "TrainableParameters",
            "model": "$model"
        }
    },
    "scheduler": {
        "_name": "ReduceLROnPlateau",
        "patience": 1,
        "mode": "min",
        "factor": 0.5,
        "optimizer": "$optimizer"
    },
    "trainer": {
        "_name": "SingleTaskTrainer",
        "model": "$model",
        "dataset_splits": "$my_dataset_splits",
        "loss": {
            "_name": "CrossEntropyLoss"
        },
        "optimizer": "$optimizer",
        "gradient_clipping": 0.25,
        "num_epochs": 5,
        "seed": 1337,
        "scheduler": "$scheduler",
        "regularizer": {
            "_name": "L1"
        },
        "metrics": {
            "accuracy": {
                "_name": "Accuracy"
            },
            "fbeta": {
                "_name": "create_fbeta"
            },
            "loss": {
                "_name": "LossMetric",
                "loss_fn": {
                    "_name": "CrossEntropyLoss"
                }
            }
        }
    }

}


class RegistryTest(unittest.TestCase):

    def test_config(self):
        e = copy.deepcopy(EXPERIMENT)
        e = ExperimentConfig(e)
        trainer = e.experiment['trainer']

        self.assertIsInstance(trainer.model, TestModel)
        self.assertIsInstance(trainer.dataset_splits, TestDataset)
        self.assertIsInstance(trainer.loss, torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        self.assertEqual(len(trainer.metrics), 3)
        self.assertEqual(trainer.device, torch.device(type='cpu'))
        self.assertEqual(trainer.seed, 1337)
        self.assertEqual(trainer.loss_accumulation_steps, 4)
        self.assertIsInstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(trainer.num_epochs, 5)
        self.assertIsInstance(trainer.regularizer, L1)
        self.assertEqual(trainer.gradient_clipping, 0.25)
        self.assertEqual(trainer.embeddings_name, None)
        self.assertEqual(trainer.forward_params, ['x_in', 'apply_softmax'])

    def test_setup(self):
        e = copy.deepcopy(EXPERIMENT)
        e = ExperimentConfig(e)
        trainer = e.experiment['trainer']
        trainer.setup(training_metrics=trainer.training_metrics)

        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.EPOCH_COMPLETED]), 6)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.ITERATION_COMPLETED]), 16)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.COMPLETED]), 2)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.STARTED]), 0)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.EPOCH_STARTED]), 8)
        self.assertEqual(len(trainer.trainer._event_handlers[ignite.engine.Events.ITERATION_STARTED]), 0)

    def test_forward(self):
        e = copy.deepcopy(EXPERIMENT)
        e = ExperimentConfig(e)
        trainer = e.experiment['trainer']
        trainer.setup(training_metrics=trainer.training_metrics)

        batch = next(iter(trainer.dataset_splits.train_data_loader()))
        self.assertEqual(list(batch.keys()), ['x_in', 'y_target'])
        output = trainer._forward(batch=batch)
        self.assertEqual(output.size()[0], min(len(trainer.dataset_splits.train_set), 128))
        self.assertEqual(output.size()[1], trainer.model.output_dim)
