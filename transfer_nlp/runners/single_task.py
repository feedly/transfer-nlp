"""
Runner class


You should define a json config file and place it into the /experiments folders
A CustomDataset class should be implemented, as well as a nn.Module, a Vectorizer and a Vocabulary (if the initial class is insufficient for the need)

This file aims at launching an experiments based on a config file

"""

import logging
from typing import Dict

import torch
from knockknock import slack_sender
from ignite.engine import Events

from transfer_nlp.runners.runnersABC import RunnerABC

name = 'transfer_nlp.runners.single_task'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


class Runner(RunnerABC):

    def __init__(self, config_args: Dict):

        super().__init__(config_args=config_args)

        # To add custom events in addition to those available in the RunnerABC class, just do:
        @self.trainer.on(Events.COMPLETED)
        def custom_event(trainer):
            logger.info("Training completed")

    def update(self, batch_dict: Dict, running_loss: float, batch_index: int, running_metrics: Dict, compute_gradient: bool = True):
        """
        If compute_gradient is True, this is a training update, otherwise this is a validation / test pass
        :param batch_dict:
        :param running_loss:
        :param batch_index:
        :param running_metrics:
        :param compute_gradient:
        :return:
        """
        if compute_gradient:
            self.optimizer.zero_grad()

        model_inputs = {inp: batch_dict[inp] for inp in self.model.inputs_names}
        y_pred = self.model(**model_inputs)

        loss_params = {
            "input": y_pred,
            "target": batch_dict['y_target']}

        if hasattr(self.loss.loss, 'mask') and self.mask_index:
            loss_params['mask_index'] = self.mask_index

        loss = self.loss.loss(**loss_params)
        penalty = torch.Tensor([0])
        if hasattr(self, "regularizer"):
            penalty += self.regularizer.regularizer.compute_penalty(model=self.model)

        loss_batch = loss.item() + penalty.item()
        # TODO: see if we can improve the online average (check exponential average)
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        if compute_gradient:
            loss.backward()  # TODO: See if we want to optimize loss or loss + penalty
            # Gradient clipping
            if hasattr(self, 'gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()

        for metric in self.metrics.names:
            metric_batch = self.metrics.metrics[metric](**loss_params)
            running_metrics[f"running_{metric}"] += (metric_batch - running_metrics[f"running_{metric}"]) / (batch_index + 1)

        return running_loss, running_metrics


slack_webhook_url = "YourWebhookURL"
slack_channel = "YourFavoriteSlackChannel"


@slack_sender(webhook_url=slack_webhook_url, channel=slack_channel)
def run_with_slack(runner, test_at_the_end: bool = False):
    runner.run(test_at_the_end=test_at_the_end)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="launch an experiment")

    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    args.config = args.config or 'experiments/mlp.json'
    runner = Runner.load_from_project(experiment_file=args.config)

    if slack_webhook_url and slack_webhook_url != "YourWebhookURL":
        run_with_slack(runner=runner, test_at_the_end=True)
    else:
        # runner.run(test_at_the_end=True)
        runner.run_pipeline()
