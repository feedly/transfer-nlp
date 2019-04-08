import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed_everywhere(seed: int, cuda: bool):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath: Path):

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_training_state(args: Dict) -> Dict:

    state = {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args['lr'],
            'epoch_index': 0,
            'train_loss': [],
            'val_loss': [],
            'test_loss': -1,
            'model_filename': args['model_state_file']}

    for metric in args['metrics']:
        state[f"train_{metric}"] = []
        state[f"val_{metric}"] = []
        state[f"test_{metric}"] = []

    return state


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


# def predict_nationality(surname: str, model: nn.Module, vectorizer: Vectorizer) -> Dict[str, Any]:
#
#     vectorized_surname = vectorizer.vectorize(surname)
#
#     if len(vectorized_surname.shape) == 1:
#         vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)
#
#     elif len(vectorized_surname.shape) == 2:
#         vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
#     else:
#         raise ValueError("The vectorized surname should be a size 1 or 2 tensor")
#
#     result = model(vectorized_surname, apply_softmax=True)
#
#     probability_values, indices = result.max(dim=1)
#     index = indices.item()
#
#     predicted_nationality = vectorizer.target_vocab.lookup_index(index)
#     probability_value = probability_values.item()
#
#     return {
#         'nationality': predicted_nationality,
#         'probability': probability_value}
#
#
# def predict_topk_nationality(surname: str, model: nn.Module, vectorizer: Vectorizer, k: int=5) -> List[Dict[str, Any]]:
#
#     vectorized_surname = vectorizer.vectorize(input_string=surname)
#
#     if len(vectorized_surname.shape) == 1:
#         vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)
#
#     elif len(vectorized_surname.shape) == 2:
#         vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
#     else:
#         raise ValueError("The vectorized surname should be a size 1 or 2 tensor")
#
#     prediction_vector = model(vectorized_surname, apply_softmax=True)
#     probability_values, indices = torch.topk(prediction_vector, k=k)
#
#     # returned size is 1,k
#     probability_values = probability_values.detach().numpy()[0]
#     indices = indices.detach().numpy()[0]
#
#     results = []
#     for prob_value, index in zip(probability_values, indices):
#         nationality = vectorizer.target_vocab.lookup_index(index)
#         results.append({'nationality': nationality, 'probability': prob_value})
#
#     return results
