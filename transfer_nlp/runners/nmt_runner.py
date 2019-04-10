import logging
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from knockknock import slack_sender

from transfer_nlp.models.nmt import NMTSampler
from transfer_nlp.plugins.generators import generate_nmt_batches
from transfer_nlp.runners.single_task import SingleTaskRunner

name = 'transfer_nlp.runners.nmt_runner'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')


class NMT_runner(SingleTaskRunner):

    def __init__(self, config_args: Dict):

        super().__init__(config_args=config_args)

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

    args.config = args.config or 'experiments/nmt.json'
    runner = SingleTaskRunner.load_from_project(experiment_file=args.config)

    if slack_webhook_url and slack_webhook_url != "YourWebhookURL":
        run_with_slack(runner=runner, test_at_the_end=False)
    else:
        runner.run(test_at_the_end=False)