import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders.vectorizers import Vectorizer


def compute_accuracy(input, target):

    if len(input.size()) == 1:  # if y_pred contains binary logits, then just compute the sigmoid to get probas
        y_pred_indices = (torch.sigmoid(input) > 0.5).cpu().long()  # .max(dim=1)[1]
    elif len(input.size()) == 2:  # then we are in the softmax case, and we take the max
        _, y_pred_indices = input.max(dim=1)
    else:
        y_pred_indices = input.max(dim=1)[1]

    n_correct = torch.eq(y_pred_indices, target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed: int, cuda: bool):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath: Path):

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_training_state(args: Dict) -> Dict:

    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args['lr'],
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args['model_state_file']}


def update_train_state(config_args: Dict, model: nn.Module, train_state: Dict):

    # Save one model at least once
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_t = train_state['val_loss'][train_state['epoch_index'] - 1]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= config_args['early_stopping_criteria']

    return train_state


def predict_nationality(surname: str, model: nn.Module, vectorizer: Vectorizer) -> Dict[str, Any]:

    vectorized_surname = vectorizer.vectorize(surname)

    if len(vectorized_surname.shape) == 1:
        vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)

    elif len(vectorized_surname.shape) == 2:
        vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
    else:
        raise ValueError("The vectorized surname should be a size 1 or 2 tensor")

    result = model(vectorized_surname, apply_softmax=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.target_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {
        'nationality': predicted_nationality,
        'probability': probability_value}


def predict_topk_nationality(surname: str, model: nn.Module, vectorizer: Vectorizer, k: int=5) -> List[Dict[str, Any]]:

    vectorized_surname = vectorizer.vectorize(input_string=surname)

    if len(vectorized_surname.shape) == 1:
        vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)

    elif len(vectorized_surname.shape) == 2:
        vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
    else:
        raise ValueError("The vectorized surname should be a size 1 or 2 tensor")

    prediction_vector = model(vectorized_surname, apply_softmax=True)
    probability_values, indices = torch.topk(prediction_vector, k=k)

    # returned size is 1,k
    probability_values = probability_values.detach().numpy()[0]
    indices = indices.detach().numpy()[0]

    results = []
    for prob_value, index in zip(probability_values, indices):
        nationality = vectorizer.target_vocab.lookup_index(index)
        results.append({'nationality': nationality, 'probability': prob_value})

    return results


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


def compute_accuracy_sequence(input, target, mask_index):
    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(input, target, mask_index):
    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)
    return F.cross_entropy(input=y_pred, target=y_true, ignore_index=mask_index)