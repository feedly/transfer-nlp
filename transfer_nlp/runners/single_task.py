"""
Runner class


You should define a json config file and place it into the /experiments folders
A CustomDataset class should be implemented, as well as a nn.Module, a Vectorizer and a Vocabulary (if the initial class is insufficient for the need)

This file aims at launching an experiments based on a config file

"""

import logging
from pathlib import Path
from typing import Dict

import torch
from knockknock import slack_sender
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, WeightsScalarHandler, WeightsHistHandler, \
    GradsScalarHandler, GradsHistHandler
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from ignite.handlers import EarlyStopping

from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.runners.runnersABC import RunnerABC
from transfer_nlp.models import *
from transfer_nlp.models.perceptrons2 import *
from transfer_nlp.experiments.surnames import *

name = 'transfer_nlp.runners.single_task'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


class Runner(RunnerABC):

    def __init__(self, config_args: Dict):

        super().__init__(config_args=config_args)

        # We show here how to add some events: tensorboard logs!
        tb_logger = TensorboardLogger(log_dir=self.config_args['logs'])
        tb_logger.attach(self.trainer,
                         log_handler=OutputHandler(tag="training", output_transform=lambda loss: {
                             'loss': loss}),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(self.evaluator,
                         log_handler=OutputHandler(tag="validation",
                                                   metric_names=["loss", "accuracy"],
                                                   another_engine=self.trainer),
                         event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(self.trainer,
                         log_handler=OptimizerParamsHandler(self.optimizer),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(self.trainer,
                         log_handler=WeightsScalarHandler(self.model),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(self.trainer,
                         log_handler=WeightsHistHandler(self.model),
                         event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(self.trainer,
                         log_handler=GradsScalarHandler(self.model),
                         event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(self.trainer,
        #                  log_handler=GradsHistHandler(self.model),
        #                  event_name=Events.EPOCH_COMPLETED)

        # This is important to close the tensorboard file logger
        @self.trainer.on(Events.COMPLETED)
        def end_tensorboard(trainer):
            logger.info("Training completed")
            tb_logger.close()

        @self.trainer.on(Events.COMPLETED)
        def log_embeddings(trainer):

            if hasattr(self.model, "embedding"):
                logger.info("Logging embeddings to Tensorboard!")
                embeddings = self.model.embedding.weight.data
                metadata = [str(self.vectorizer.data_vocab._id2token[token_index]).encode('utf-8') for token_index in range(embeddings.shape[0])]
                self.writer.add_embedding(mat=embeddings, metadata=metadata, global_step=self.trainer.state.epoch)

            if hasattr(self.model, "entity_embedding"):
                logger.info("Logging entities embeddings to Tensorboard!")
                embeddings = self.model.entity_embedding.weight.data
                metadata = [str(self.vectorizer.target_vocab._id2token[token_index]).encode('utf-8') for token_index in range(embeddings.shape[0])]
                self.writer.add_embedding(mat=embeddings, metadata=metadata, global_step=self.trainer.state.epoch)

        handler = ModelCheckpoint(dirname=self.config_args['save_dir'], filename_prefix='experiment', save_interval=2, n_saved=2, create_dir=True, require_empty=False)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
            'mymodel': self.model})

        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=self.trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        # Terminate if NaNs are created after an iteration
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


slack_webhook_url = "YourWebhookURL"
slack_channel = "YourFavoriteSlackChannel"


@slack_sender(webhook_url=slack_webhook_url, channel=slack_channel)
def run_with_slack(runner, test_at_the_end: bool = False):
    runner.run(test_at_the_end=test_at_the_end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser(description="launch an experiment")

    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    args.config = args.config or 'experiments/newsClassifier.json'

    # runner = Runner.load_from_project(experiment_file=args.config, HOME=str(Path.home()))
    # runner.run_pipeline()
    # experiment = ExperimentConfig.from_json('/Users/kireet/git/transfer-nlp/transfer_nlp/experiments/mlp.json', HOME=str(Path.home()))
    path = '/Users/petermartigny/Documents/PycharmProjects/transfer-nlp/transfer_nlp/experiments/mlp.json'
    experiment = ExperimentConfig.from_json(path, HOME=str(Path.home()))
    # experiment['trainer'].train()
    #
    # if slack_webhook_url and slack_webhook_url != "YourWebhookURL":
    #     run_with_slack(runner=runner, test_at_the_end=True)
    # else:
    #     # runner.run(test_at_the_end=True)
    #     runner.run_pipeline()
