import logging
from typing import Dict

import torch
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor

name = 'transfer_nlp.runners.trainers'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)

def _prepare_batch(batch: Dict, device=None, non_blocking: bool=False):
    """Prepare batch for training: pass to a device with options.

    """
    result = {key: convert_tensor(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
    return result


def create_supervised_trainer(experiment, prepare_batch=_prepare_batch, non_blocking=False):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if experiment.config_args['device']:
        experiment.model.to(experiment.config_args['device'])

    def _update(engine, batch):

        experiment.model.train()
        experiment.optimizer.zero_grad()
        batch = prepare_batch(batch, device=experiment.config_args['device'], non_blocking=non_blocking)
        model_inputs = {inp: batch[inp] for inp in experiment.model.inputs_names}
        y_pred = experiment.model(**model_inputs)

        loss_params = {
            "input": y_pred,
            "target": batch['y_target']}

        if hasattr(experiment.loss.loss, 'mask') and experiment.mask_index:
            loss_params['mask_index'] = experiment.mask_index

        loss = experiment.loss.loss(**loss_params)
        loss.backward()
        experiment.optimizer.step()

        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(experiment, metrics: Dict, prepare_batch=_prepare_batch, non_blocking=False):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    if experiment.config_args['device']:
        experiment.model.to(experiment.config_args['device'])

    def _inference(engine, batch):
        experiment.model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=experiment.config_args['device'], non_blocking=non_blocking)
            model_inputs = {inp: batch[inp] for inp in experiment.model.inputs_names}
            y_pred = experiment.model(**model_inputs)
            return y_pred, batch['y_target']

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
