"""
This file shows an example of loss function implementation.
To implement your own loss class, you should implement your own loss function, and then create a
Loss class with the __call__ method using your loss function.

Using the decorator @register_loss allows the framework to reuse your custom loss function
"""


import torch.nn.functional as F

from transfer_nlp.plugins.metrics import normalize_sizes
from transfer_nlp.plugins.registry import register_loss

def sequence_loss(input, target, mask_index):

    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)
    return F.cross_entropy(input=y_pred, target=y_true, ignore_index=mask_index)


@register_loss
class SequenceLoss:

    def __init__(self):
        self.mask: bool = True

    def __call__(self, *args, **kwargs):
        return sequence_loss(*args, **kwargs)