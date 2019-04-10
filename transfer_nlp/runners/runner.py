"""
Runner class


You should define a json config file and place it into the /experiments folders
A CustomDataset class should be implemented, as well as a nn.Module, a Vectorizer and a Vocabulary (if the initial class is insufficient for the need)

This file aims at launching an experiments based on a config file

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from knockknock import slack_sender

from transfer_nlp.models.nmt import NMTSampler
from transfer_nlp.plugins.generators import generate_nmt_batches
from transfer_nlp.runners.runnersABC import RunnerABC
from transfer_nlp.runners.utils import update_train_state

name = 'transfer_nlp.runners.runner'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class Runner(RunnerABC):

    def __init__(self, config_args: Dict):

        super().__init__(config_args=config_args)

    def train_one_epoch(self):

        self.epoch_index += 1
        sample_probability = (20 + self.epoch_index) / self.config_args['num_epochs']  # TODO: include this into the NMT training part

        self.training_state['epoch_index'] += 1

        # Set the dataset object to train mode such that the dataset used is the training data
        self.dataset.set_split(split='train')
        batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])

        running_loss = 0
        running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}

        # Set the model object to train mode (torch optimizes the parameters)
        self.model.train()

        num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 1 == 0:
                logger.info(f"Training batch {batch_index + 1} / {num_batch}")
            self.optimizer.zero_grad()

            model_inputs = {inp: batch_dict[inp] for inp in self.model_inputs}
            y_pred = self.model(**model_inputs)

            loss_params = {
                "input": y_pred,
                "target": batch_dict['y_target']}

            if hasattr(self.loss.loss, 'mask') and self.mask_index:
                loss_params['mask_index'] = self.mask_index

            loss = self.loss.loss(**loss_params)
            penalty = 0

            # Add regularization for training part only
            for name, parameter in self.model.named_parameters():
                penalty += self.regularizer.regularizer(parameter)

            loss_batch = loss.item() + penalty.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config_args['gradient_clipping'])

            self.optimizer.step()

            for metric in self.metrics.names:
                metric_batch = self.metrics.metrics[metric](**loss_params)
                running_metrics[f"running_{metric}"] += (metric_batch - running_metrics[f"running_{metric}"]) / (batch_index + 1)

        self.training_state['train_loss'].append(running_loss)
        for metric in self.metrics.names:
            self.training_state[f"train_{metric}"].append(running_metrics[f"running_{metric}"])

        # Iterate over validation dataset
        self.dataset.set_split(split='val')
        batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])

        running_loss = 0
        running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}

        # Set the model object to val mode (torch does not optimize the parameters)
        self.model.eval()

        num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 30 == 0:
                logger.info(f"Validation batch {batch_index + 1} / {num_batch}")

            model_inputs = {inp: batch_dict[inp] for inp in self.model_inputs}
            y_pred = self.model(**model_inputs)

            loss_params = {
                "input": y_pred,
                "target": batch_dict['y_target']}

            if hasattr(self.loss.loss, 'mask') and self.mask_index:
                loss_params['mask_index'] = self.mask_index
            loss = self.loss.loss(**loss_params)
            # Add regularization for training part only
            penalty = 0
            for name, parameter in self.model.named_parameters():
                penalty += self.regularizer.regularizer(parameter)
            loss_batch = loss.item() + penalty.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            for metric in self.metrics.names:
                metric_batch = self.metrics.metrics[metric](**loss_params)
                running_metrics[f"running_{metric}"] += (metric_batch - running_metrics[f"running_{metric}"]) / (batch_index + 1)

        self.training_state['val_loss'].append(running_loss)

        for metric in self.metrics.names:
            self.training_state[f"val_{metric}"].append(running_metrics[f"running_{metric}"])

        self.training_state = update_train_state(config_args=self.config_args, model=self.model,
                                                 train_state=self.training_state)
        self.scheduler.scheduler.step(self.training_state['val_loss'][-1])

    def do_test(self):

        self.dataset.set_split(split='test')
        batch_generator = self.generator.generator(dataset=self.dataset, batch_size=self.config_args['batch_size'], device=self.config_args['device'])
        num_batch = self.dataset.get_num_batches(batch_size=self.config_args['batch_size'])

        running_loss = 0
        running_metrics = {f"running_{metric}": 0 for metric in self.metrics.names}
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 30 == 0:
                logger.info(f"Test batch {batch_index + 1} / {num_batch}")

            model_inputs = {inp: batch_dict[inp] for inp in self.model_inputs}
            y_pred = self.model(**model_inputs)

            loss_params = {
                "input": y_pred,
                "target": batch_dict['y_target']}
            if hasattr(self.loss.loss, 'mask') and self.mask_index:
                loss_params['mask_index'] = self.mask_index

            loss = self.loss.loss(**loss_params)

            penalty = 0
            for name, parameter in self.model.named_parameters():
                penalty += self.regularizer.regularizer(parameter)
            loss_batch = loss.item() + penalty.item()

            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            for metric in self.metrics.names:
                metric_batch = self.metrics.metrics[metric](**loss_params)
                running_metrics[f"running_{metric}"] += (metric_batch - running_metrics[f"running_{metric}"]) / (batch_index + 1)

        self.training_state['test_loss'] = running_loss
        for metric in self.metrics.names:
            self.training_state[f"test_{metric}"].append(running_metrics[f"running_{metric}"])

    # The following methods are for NMT only #TODO: clean this part
    def visualize_nmt_test(self):

        model = self.model.eval().to(self.args.device)
        sampler = NMTSampler(vectorizer=self.vectorizer, model=model)

        self.dataset.set_split('test')
        batch_generator = generate_nmt_batches(dataset=self.dataset,
                                               batch_size=self.args.batch_size,
                                               device=self.args.device)
        test_results = []
        for batch_dict in batch_generator:
            sampler.apply_to_batch(batch_dict)
            for i in range(self.args.batch_size):
                test_results.append(sampler.get_ith_item(i, False))

        plt.hist([r['bleu-4'] for r in test_results], bins=100)
        plt.show()
        average = np.mean([r['bleu-4'] for r in test_results])
        median = np.median([r['bleu-4'] for r in test_results])
        logger.info(f"Average Bleu: {average}")
        logger.info(f"Median Bleu: {median}")

    def get_best(self) -> List[Dict[str, Any]]:

        self.dataset.set_split('val')
        batch_generator = generate_nmt_batches(dataset=self.dataset,
                                               batch_size=self.args.batch_size,
                                               device=self.args.device)
        batch_dict = next(batch_generator)

        model = self.model.eval().to(self.args.device)
        sampler = NMTSampler(self.vectorizer, model)
        sampler.apply_to_batch(batch_dict)
        all_results = []
        for i in range(self.args.batch_size):
            all_results.append(sampler.get_ith_item(i, False))
        top_results = [x for x in all_results if x['bleu-4'] > 0.1]

        return top_results

    def visualize_results(self):

        top_results = self.get_best()

        for sample in top_results:
            plt.figure()
            target_len = len(sample['sampled'])
            source_len = len(sample['source'])

            attention_matrix = sample['attention'][:target_len, :source_len + 2].transpose()  # [::-1]
            ax = sns.heatmap(attention_matrix, center=0.0)
            ylabs = ["<BOS>"] + sample['source'] + ["<EOS>"]
            # ylabs = sample['source']
            # ylabs = ylabs[::-1]
            ax.set_yticklabels(ylabs, rotation=0)
            ax.set_xticklabels(sample['sampled'], rotation=90)
            ax.set_xlabel("Target Sentence")
            ax.set_ylabel("Source Sentence\n\n")
            plt.show()


def run_experiment(experiment_file: str):
    """
    Instantiate an experiment
    :param experiment_file:
    :return:
    """

    experiments_path = Path(__file__).resolve().parent.parent
    experiments_path /= experiment_file

    with open(experiments_path, 'r') as exp:
        experiment = json.load(exp)

    runner = Runner(config_args=experiment)
    # runner.run()
    return runner


slack_webhook_url = "YourWebhookURL"
slack_channel = "YourFavoriteSlackChannel"


@slack_sender(webhook_url=slack_webhook_url, channel=slack_channel)
def run_with_slack(runner, test_at_the_end: bool = False):
    runner.run(test_at_the_end=test_at_the_end)


if __name__ == "__main__":
    import argparse

    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="launch an experiment")

    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    args.config = args.config or 'experiments/mlp.json'
    runner = run_experiment(experiment_file=args.config)

    if slack_webhook_url and slack_webhook_url != "YourWebhookURL":
        run_with_slack(runner=runner, test_at_the_end=False)
    else:
        runner.run(test_at_the_end=False)

    # surname = "Mueller"
    # surnames = ["Zhang", "McDonald", "Dupont", "Okinawa", "Mueller"]
    #
    # for surname in surnames:
    #     prediction = predict_nationality(surname=surname, model=runner.model, vectorizer=runner.vectorizer)
    #     print(f"{surname} --> {prediction}")
