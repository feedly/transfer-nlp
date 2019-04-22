import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from smart_open import open
from tqdm import tqdm

from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.experiments.news import NewsDatasetSplits

name = 'transfer_nlp.runners.single_task'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


def load_glove_from_file(glove_filepath: Path) -> Tuple[Dict[str, int], np.array]:

    w2i = {}
    embeddings = []

    with open(glove_filepath, "r") as fp:

        for index, line in tqdm(enumerate(fp), "Embeddings"):
            line = line.split(" ")  # each line: word num1 num2 ...
            w2i[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)

    return w2i, np.stack(embeddings)

@register_plugin
class EmbeddingsHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: NewsDatasetSplits):
        super().__init__()
        self.words = dataset_splits.vectorizer.data_vocab._token2id.keys()


@register_plugin
class Embedding:

    def __init__(self, glove_filepath: Path, embedding_hyper_params: ObjectHyperParams):

        words = embedding_hyper_params.words

        w2i, glove_embeddings = load_glove_from_file(glove_filepath=glove_filepath)
        embedding_size = glove_embeddings.shape[1]

        final_embeddings = np.zeros((len(words), embedding_size))

        for i, word in tqdm(enumerate(words), total=len(words), desc='Loading pre-trained word embeddings'):
            if word in w2i:
                final_embeddings[i, :] = glove_embeddings[w2i[word]]
            else:
                embedding_i = torch.ones(1, embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i)
                final_embeddings[i, :] = embedding_i

        self.embeddings = final_embeddings

