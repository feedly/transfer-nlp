"""
Runner class


You should define a json config file and place it into the /experiments folders
A CustomDataset class should be implemented, as well as a nn.Module, a Vectorizer and a Vocabulary (if the initial class is insufficient for the need)

This file aims at launching an experiments based on a config file

"""


import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from embeddings.embeddings import make_embedding_matrix
from embeddings.utils import pretty_print, get_closest
from loaders.loaders import ReviewsDataset, SurnamesDataset, generate_batches, generate_nmt_batches, SurnamesDatasetCNN, CBOWDataset, NewsDataset, \
    CustomDataset, SurnameDatasetRNN, \
    SurnameDatasetGeneration, NMTDataset, FeedlyDataset
from loaders.vectorizers import Vectorizer
from models.cbow import CBOWClassifier
from models.cnn import SurnameClassifierCNN, NewsClassifier, predict_category
from models.generation import SurnameConditionedGenerationModel, decode_samples, generate_names
from models.nmt import NMTModel, NMTSampler
from models.perceptrons import MultiLayerPerceptron, Perceptron, predict_review, inspect_model, preprocess
from models.rnn import SurnameClassifierRNN, predict_nationalityRNN
from runners.utils import compute_accuracy, set_seed_everywhere, handle_dirs, make_training_state, update_train_state, predict_nationality, \
    predict_topk_nationality, sequence_loss, compute_accuracy_sequence

name = 'transfer_nlp.runners.runner'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)
logging.info('')

DATASET_CLASSES = {'NewsDataset': NewsDataset,
                   'CBOWDataset': CBOWDataset,
                   'SurnamesDatasetCNN': SurnamesDatasetCNN,
                   'SurnamesDataset': SurnamesDataset,
                   'ReviewsDataset': ReviewsDataset,
                   'SurnameDatasetRNN': SurnameDatasetRNN,
                   'SurnameDatasetGeneration': SurnameDatasetGeneration,
                   'NMTDataset': NMTDataset,
                   'FeedlyDataset': FeedlyDataset}

MODEL_CLASSES = {'NewsClassifier': NewsClassifier,
                 'CBOWClassifier': CBOWClassifier,
                 'SurnameClassifierCNN': SurnameClassifierCNN,
                 'MultiLayerPerceptron': MultiLayerPerceptron,
                 'Perceptron': Perceptron,
                 'SurnameClassifierRNN': SurnameClassifierRNN,
                 'SurnameConditionedGenerationModel': SurnameConditionedGenerationModel,
                 'NMTModel': NMTModel}

UTILS_FUNCTIONS = [preprocess, inspect_model, predict_review, predict_category, get_closest, pretty_print, predict_nationality,
                   predict_topk_nationality, decode_samples, predict_nationalityRNN, generate_names]

class Runner:

    def __init__(self, args):

        self.dataset_cls = DATASET_CLASSES[args.dataset_cls]
        self.model: nn.Module = MODEL_CLASSES[args.model]

        self.args: Namespace = args
        self.dataset: CustomDataset = None
        self.training_state: Dict = {}
        self.vectorizer: Vectorizer = None
        self.loss_func: nn.modules.loss._Loss = None
        self.optimizer: optim.optimizer.Optimizer = None
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = None
        self.is_output_continuous = True
        self.is_pred_continuous = True
        self.mask_index: int = None
        self.epoch_index: int = 0
        self.writer = SummaryWriter(log_dir=args.logs)

        self.instantiate()

    def instantiate(self):

        if self.args.expand_filepaths_to_save_dir:
            self.args.vectorizer_file = self.args.save_dir + '/' + self.args.vectorizer_file

            self.args.model_state_file = self.args.save_dir + '/' + self.args.model_state_file

            logger.info("Expanded filepaths: ")
            logger.info(f"{self.args.vectorizer_file}")
            logger.info(f"{self.args.model_state_file}")

        self.training_state = make_training_state(args=self.args)

        if not torch.cuda.is_available():
            self.args.cuda = False
        self.args.device = torch.device("cuda" if self.args.cuda else "cpu")

        # Set seed for reproducibility
        set_seed_everywhere(self.args.seed, self.args.cuda)

        # handle dirs
        handle_dirs(self.args.save_dir)

        # Load dataset and vectorizer
        logger.info("Loading the data and getting the vectorizer ready")

        if self.args.reload_from_files:
            # training from a checkpoint
            logger.info("Loading dataset and vectorizer")
            self.dataset = self.dataset_cls.load_dataset_and_load_vectorizer(self.args.data_csv,
                                                                        self.args.vectorizer_file)
        else:
            logger.info("Loading dataset and creating vectorizer")
            # create dataset and vectorizer
            self.dataset = self.dataset_cls.load_dataset_and_make_vectorizer(self.args.data_csv)
            self.dataset.save_vectorizer(self.args.vectorizer_file)

        self.vectorizer = self.dataset.get_vectorizer()

        ##### Instantiate classifier #####

        # Use GloVe or randomly initialized embeddings
        if self.args.use_glove:
            words = self.vectorizer.data_vocab._token2id.keys()
            embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                               words=words)
            logging.info("Using pre-trained embeddings")
        else:
            logger.info("Not using pre-trained embeddings")
            embeddings = None

        if self.model == Perceptron:
            self.model = Perceptron(num_features=len(self.vectorizer.data_vocab))  # 1 1-layer perceptron
        elif self.model == MultiLayerPerceptron:
            self.model = MultiLayerPerceptron(input_dim=len(self.vectorizer.data_vocab), hidden_dim=self.args.hidden_dim,
                                              output_dim=len(self.vectorizer.target_vocab))
            # self.model = MultiLayerPerceptron(input_dim=len(self.vectorizer.data_vocab), hidden_dim=self.args.hidden_dim, output_dim=1)  #2 MLP
        elif self.model == SurnameClassifierCNN:
            self.model = SurnameClassifierCNN(initial_num_channels=len(self.vectorizer.data_vocab),
                                              num_classes=len(self.vectorizer.target_vocab),
                                              num_channels=self.args.num_channels)
        elif self.model == CBOWClassifier:
            self.model = CBOWClassifier(vocabulary_size=len(self.vectorizer.data_vocab),
                                        embedding_size=self.args.embedding_size)
        elif self.model == NewsClassifier:
            self.model = NewsClassifier(embedding_size=self.args.embedding_size,
                                             num_embeddings=len(self.vectorizer.data_vocab),
                                             num_channels=self.args.num_channels,
                                             hidden_dim=self.args.hidden_dim,
                                             num_classes=len(self.vectorizer.target_vocab),
                                             dropout_p=self.args.dropout_p,
                                             pretrained_embeddings=embeddings,
                                             padding_idx=0)
        elif self.model == SurnameClassifierRNN:
            self.model = SurnameClassifierRNN(embedding_size=self.args.char_embedding_size,
                               num_embeddings=len(self.vectorizer.data_vocab),
                               num_classes=len(self.vectorizer.target_vocab),
                               rnn_hidden_size=self.args.rnn_hidden_size,
                               padding_idx=self.vectorizer.data_vocab.mask_index)

        elif self.model == SurnameConditionedGenerationModel:
            self.model = SurnameConditionedGenerationModel(char_embedding_size=self.args.char_embedding_size,
                                   char_vocab_size=len(self.vectorizer.data_vocab),
                                   num_nationalities=len(self.vectorizer.target_vocab),
                                   rnn_hidden_size=self.args.rnn_hidden_size,
                                   padding_idx=self.vectorizer.data_vocab.mask_index,
                                   dropout_p=0.5,
                                   conditioned=self.args.conditioned)
            self.mask_index = self.vectorizer.data_vocab.mask_index
            self.is_pred_continuous = False

        elif self.model == NMTModel:

            self.model = NMTModel(source_vocab_size=len(self.vectorizer.data_vocab),
                             source_embedding_size=self.args.source_embedding_size,
                             target_vocab_size=len(self.vectorizer.target_vocab),
                             target_embedding_size=self.args.target_embedding_size,
                             encoding_size=self.args.encoding_size,
                             target_bos_index=self.vectorizer.target_vocab.begin_seq_index)
            self.mask_index = self.vectorizer.data_vocab.mask_index
            self.is_pred_continuous = False

        else:
            logger.info("You must first design a model and then use it as argument")


        logger.info("Using the following classifier:")
        logger.info(f"{self.model}")
        self.model = self.model.to(self.args.device)

        # Define loss function and optimizer
        if self.dataset_cls == ReviewsDataset:
            self.loss_func = nn.BCEWithLogitsLoss()

        elif self.dataset_cls in [SurnamesDataset, SurnamesDatasetCNN, NewsDataset, SurnameDatasetRNN]:
            self.dataset.class_weights = self.dataset.class_weights.to(self.args.device)
            self.loss_func = nn.CrossEntropyLoss(weight=self.dataset.class_weights)
            self.is_output_continuous = False  # This is for the runner to take into account different loss functions, whether they ouput logits or not

        elif self.dataset_cls == CBOWDataset:
            self.loss_func = nn.CrossEntropyLoss()
            self.is_output_continuous = False
            self.is_pred_continuous = False

        elif self.dataset_cls in [SurnameDatasetGeneration, NMTDataset, FeedlyDataset]:
            pass

        else:
            raise ValueError("Only Yelp Reviews, Surnames and CBOW are available at the moment")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)

        if self.dataset_cls == NewsDataset:
            self.is_output_continuous = False
            self.is_pred_continuous = False

    def train_one_epoch(self):

        self.epoch_index += 1
        sample_probability = (20 + self.epoch_index) / self.args.num_epochs

        self.training_state['epoch_index'] += 1

        # Set the dataset object to train mode such that the dataset used is the training data
        self.dataset.set_split(split='train')

        if isinstance(self.dataset, NMTDataset):
            batch_generator = generate_nmt_batches(dataset=self.dataset,
                                                   batch_size=self.args.batch_size,
                                                   device=self.args.device)

        else:
            batch_generator = generate_batches(data=self.dataset, batch_size=self.args.batch_size, device=self.args.device)

        running_loss = 0
        running_acc = 0

        # Set the model object to train mode (torch optimizes the parameters)
        self.model.train()

        num_batch = self.dataset.get_num_batches(batch_size=self.args.batch_size)

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 1 == 0:
                logger.info(f"Training batch {batch_index + 1} / {num_batch}")
            self.optimizer.zero_grad()

            if isinstance(self.model, SurnameConditionedGenerationModel):
                y_pred = self.model(x_in=batch_dict['x_data'],
                               nationality_index=batch_dict['class_index'])

            elif isinstance(self.model, SurnameClassifierRNN):
                y_pred = self.model(x_in=batch_dict['x_data'],
                                    x_lengths=batch_dict['x_length'])
            elif isinstance(self.model, NMTModel):
                y_pred = self.model(batch_dict['x_source'],
                               batch_dict['x_source_length'],
                               batch_dict['x_target'],
                               sample_probability=sample_probability)
            else:
                if self.is_pred_continuous:
                    y_pred = self.model(x_in=batch_dict['x_data'].float())
                else:
                    y_pred = self.model(x_in=batch_dict['x_data'])

            if self.mask_index:
                loss = sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)
            else:
                if self.is_output_continuous:
                    loss = self.loss_func(y_pred, batch_dict['y_target'].float())
                else:
                    loss = self.loss_func(y_pred, batch_dict['y_target'])

            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)

            self.optimizer.step()

            if self.mask_index:
                acc_batch = compute_accuracy_sequence(y_pred=y_pred, y_true=batch_dict['y_target'], mask_index=self.mask_index)
            else:
                acc_batch = compute_accuracy(y_pred=y_pred, y_target=batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        self.training_state['train_loss'].append(running_loss)
        self.training_state['train_acc'].append(running_acc)

        # Iterate over validation dataset
        self.dataset.set_split(split='val')

        if isinstance(self.dataset, NMTDataset):
            batch_generator = generate_nmt_batches(dataset=self.dataset,
                                                   batch_size=self.args.batch_size,
                                                   device=self.args.device)

        else:
            batch_generator = generate_batches(data=self.dataset, batch_size=self.args.batch_size, device=self.args.device)

        running_loss = 0
        running_acc = 0
        # Set the model object to val mode (torch does not optimize the parameters)
        self.model.eval()

        num_batch = self.dataset.get_num_batches(batch_size=self.args.batch_size)

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 30 == 0:
                logger.info(f"Validation batch {batch_index + 1} / {num_batch}")

            if isinstance(self.model, SurnameConditionedGenerationModel):
                y_pred = self.model(x_in=batch_dict['x_data'],
                               nationality_index=batch_dict['class_index'])

            elif isinstance(self.model, SurnameClassifierRNN):
                y_pred = self.model(x_in=batch_dict['x_data'],
                                    x_lengths=batch_dict['x_length'])

            elif isinstance(self.model, NMTModel):
                y_pred = self.model(batch_dict['x_source'],
                               batch_dict['x_source_length'],
                               batch_dict['x_target'],
                               sample_probability=sample_probability)

            else:
                if self.is_pred_continuous:
                    y_pred = self.model(x_in=batch_dict['x_data'].float())
                else:
                    y_pred = self.model(x_in=batch_dict['x_data'])

            if self.mask_index:
                loss = sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)
            else:
                if self.is_output_continuous:
                    loss = self.loss_func(y_pred, batch_dict['y_target'].float())
                else:
                    loss = self.loss_func(y_pred, batch_dict['y_target'])

            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            if self.mask_index:
                acc_batch = compute_accuracy_sequence(y_pred=y_pred, y_true=batch_dict['y_target'], mask_index=self.mask_index)
            else:
                acc_batch = compute_accuracy(y_pred=y_pred, y_target=batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        self.training_state['val_loss'].append(running_loss)
        self.training_state['val_acc'].append(running_acc)
        self.training_state = update_train_state(args=self.args, model=self.model,
                                                 train_state=self.training_state)
        self.scheduler.step(self.training_state['val_loss'][-1])

    def do_test(self):

        self.dataset.set_split(split='test')
        num_batch = self.dataset.get_num_batches(batch_size=self.args.batch_size)

        if isinstance(self.dataset, NMTDataset):
            batch_generator = generate_nmt_batches(dataset=self.dataset,
                                                   batch_size=self.args.batch_size,
                                                   device=self.args.device)

        else:
            batch_generator = generate_batches(data=self.dataset, batch_size=self.args.batch_size, device=self.args.device)

        running_loss = 0
        running_acc = 0
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            if batch_index % 30 == 0:
                logger.info(f"Test batch {batch_index + 1} / {num_batch}")

            if isinstance(self.model, SurnameConditionedGenerationModel):
                y_pred = self.model(x_in=batch_dict['x_data'],
                               nationality_index=batch_dict['class_index'])

            elif isinstance(self.model, SurnameClassifierRNN):
                y_pred = self.model(x_in=batch_dict['x_data'],
                                    x_lengths=batch_dict['x_length'])

            elif isinstance(self.model, NMTModel):
                y_pred = self.model(batch_dict['x_source'],
                               batch_dict['x_source_length'],
                               batch_dict['x_target'])

            else:
                if self.is_pred_continuous:
                    y_pred = self.model(x_in=batch_dict['x_data'].float())
                else:
                    y_pred = self.model(x_in=batch_dict['x_data'])

            if self.mask_index:
                loss = sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)
            else:
                if self.is_output_continuous:
                    loss = self.loss_func(y_pred, batch_dict['y_target'].float())
                else:
                    loss = self.loss_func(y_pred, batch_dict['y_target'])

            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            if self.mask_index:
                acc_batch = compute_accuracy_sequence(y_pred=y_pred, y_true=batch_dict['y_target'], mask_index=self.mask_index)
            else:
                acc_batch = compute_accuracy(y_pred=y_pred, y_target=batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        self.training_state['test_loss'] = running_loss
        self.training_state['test_acc'] = running_acc

    def run(self):
        """
        Training loop
        :return:
        """

        # Train/Val loop
        logger.info("#" * 50)
        logger.info("Entering the training loop...")
        logger.info("#" * 50)

        try:

            for epoch in range(self.args.num_epochs):

                logger.info("#" * 50)
                logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                logger.info("#" * 50)
                self.train_one_epoch()

                self.writer.add_scalar('Train/acc', self.training_state['train_acc'][-1], epoch)
                self.writer.add_scalar('Train/loss', self.training_state['train_loss'][-1], epoch)
                self.writer.add_scalar('Val/acc', self.training_state['val_acc'][-1], epoch)
                self.writer.add_scalar('Val/loss', self.training_state['val_loss'][-1], epoch)
                tp = {
                    "tl": self.training_state['train_loss'][-1],
                    'ta': self.training_state['train_acc'][-1],
                    'vl': self.training_state['val_loss'][-1],
                    'va': self.training_state['val_acc'][-1]}
                tp = {key: np.round(value, 3) for key, value in tp.items()}
                logger.info(f"Epoch {epoch}: train loss: {tp['tl']} / val loss: {tp['vl']} / train acc: {tp['ta']} / val acc: {tp['va']}")

                if self.training_state['stop_early']:
                    break

        except KeyboardInterrupt:
            logger.info("Leaving training phase early (Action taken by user)")

        # Test phase
        logger.info("#" * 50)
        logger.info("Entering the test phase...")
        logger.info("#" * 50)
        self.do_test()
        logger.info(f"test loss: {self.training_state['test_loss']} / test acc: {self.training_state['test_acc']}")

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


def run_experiment(config: str):

    experiments_path = Path(__file__).resolve().parent.parent / 'experiments'
    experiments_path /= config

    with open(experiments_path, 'r') as exp:
        experiment = json.load(exp)

    experiment = Namespace(**experiment)
    runner = Runner(args=experiment)
    runner.run()
    return runner


if __name__ == "__main__":
    import argparse

    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="launch an experiment")

    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    args.config = args.config or 'surnamesGeneration.json'
    runner = run_experiment(config=args.config)
    generate_names(model=runner.model, vectorizer=runner.vectorizer, character=True)
    # runner.visualize_nmt_test()
    # runner.visualize_results()

    # generate_names(model=runner.model, vectorizer=runner.vectorizer, character=False)


    # classifier = runner.model.to("cpu")
    # for surname in ['McMahan', 'Nakamoto', 'Wan', 'Cho']:
    #     logger.info(predict_nationalityRNN(surname=surname, classifier=classifier, vectorizer=runner.vectorizer))

    # runner = run_experiment(config='perceptron.json')
    # review = "This book is amazing!"
    # predicted_rating = predict_review(review=review, model=runner.model, vectorizer=runner.vectorizer)
    # logger.info(f"Review: {review} --> {predicted_rating}")
    # inspect_model(model=runner.model, vectorizer=runner.vectorizer)


    # runner = run_experiment(config='mlp.json')
    # surnames = ["McDonald", "Aleksander", "Mahmoud", "Zhang", "Dupont", "Rastapopoulos"]
    # for surname in surnames:
    #     print(surname)
    #     print(predict_nationality(surname=surname, model=runner.model, vectorizer=runner.vectorizer))
    # predict_topk_nationality(surname="Zhang", model=runner.model, vectorizer=runner.vectorizer, k=10)


    # runner = run_experiment(config='surnameClassifier.json')
    # surnames = ["McDonald", "Aleksander", "Mahmoud", "Zhang", "Dupont", "Rastapopoulos"]
    # for surname in surnames:
    #     print(surname)
    #     print(predict_nationality(surname=surname, model=runner.model, vectorizer=runner.vectorizer))
    # predict_topk_nationality(surname="Zhang", model=runner.model, vectorizer=runner.vectorizer, k=10)


    # runner = run_experiment(config='cbow.json')
    # embeddings = runner.model.embedding.weight.data
    # word_to_idx = runner.vectorizer.data_vocab._token2id
    #
    # target_words = ['frankenstein', 'monster', 'science', 'sickness', 'lonely', 'happy']
    #
    # embeddings = runner.model.embedding.weight.data
    # word_to_idx = runner.vectorizer.data_vocab._token2id
    #
    # for target_word in target_words:
    #     print(f"======={target_word}=======")
    #     if target_word not in word_to_idx:
    #         print("Not in vocabulary")
    #         continue
    #     pretty_print(get_closest(target_word=target_word, word_to_idx=word_to_idx, embeddings=embeddings, n=5))


    # runner = run_experiment(config='newsClassifier.json')
    # title = "This article is about business"
    # predict_category(title=title, model=runner.model, vectorizer=runner.vectorizer, max_length=runner.dataset._max_seq_length + 1)
