"""
This file shows an example of loss function implementation.
To implement your own loss class, you should implement your own loss function, and then create a
Loss class with the __call__ method using your loss function.

Using the decorator @register_loss allows the framework to reuse your custom loss function
"""

import torch.nn.functional as F

from transfer_nlp.loaders.loaders import DatasetSplits
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.plugins.metrics import normalize_sizes


def sequence_loss(input, target, mask_index):
    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)
    return F.cross_entropy(input=y_pred, target=y_true, ignore_index=mask_index)


@register_plugin
class SequenceLossHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: DatasetSplits):
        super().__init__()
        self.mask_index = dataset_splits.vectorizer.data_vocab.mask_index


@register_plugin
class SequenceLoss:

    def __init__(self, loss_hyper_params: SequenceLossHyperParams):
        self.mask_index = loss_hyper_params.mask_index

    def __call__(self, *args, **kwargs):
        return sequence_loss(*args, **kwargs, mask_index=self.mask_index)
