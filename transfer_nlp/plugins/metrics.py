"""
This file shows an example of metric implementation.
To implement your own metric method, you should use the decorator @register_metric which allows the framework to
reuse your custom metric method
"""

from typing import Tuple

import torch
from ignite.metrics import Loss

from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.registry import register_metric

@register_plugin
class LossMetric(Loss):
    """
    avoid name collision on batch size param of super class
    """
    def __init__(self, loss_fn):
        super().__init__(loss_fn)

def normalize_sizes(y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor sizes

    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


@register_metric
def compute_accuracy_sequence(input, target, mask_index):
    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


@register_metric
def compute_accuracy(input, target):
    if len(input.size()) == 1:  # if y_pred contains binary logits, then just compute the sigmoid to get probas
        y_pred_indices = (torch.sigmoid(input) > 0.5).cpu().long()  # .max(dim=1)[1]
    elif len(input.size()) == 2:  # then we are in the softmax case, and we take the max
        _, y_pred_indices = input.max(dim=1)
    else:
        y_pred_indices = input.max(dim=1)[1]

    n_correct = torch.eq(y_pred_indices, target).sum().item()
    return n_correct / len(y_pred_indices) * 100
