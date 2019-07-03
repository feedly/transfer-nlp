import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from smart_open import open

from transfer_nlp.loaders.loaders import DatasetSplits
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams

logger = logging.getLogger(__name__)


TQDM = True
try:
    from tqdm import tqdm
except ImportError:
    logger.debug("To use tqdm in the embedding loader, pip install tqdm")
    TQDM = False


def load_glove_from_file(glove_filepath: Path) -> Tuple[Dict[str, int], np.array]:
    w2i = {}
    embeddings = []

    with open(glove_filepath, "r") as fp:
        iterator = tqdm(enumerate(fp), "Embeddings") if TQDM else enumerate(fp)
        for index, line in iterator:
            line = line.split(" ")  # each line: word num1 num2 ...
            w2i[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)

    return w2i, np.stack(embeddings)


@register_plugin
class EmbeddingsHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: DatasetSplits):
        super().__init__()
        self.words = dataset_splits.vectorizer.data_vocab._token2id.keys()


@register_plugin
class Embedding:

    def __init__(self, glove_filepath: Union[Path, str], data: DatasetSplits):

        words = data.vectorizer.data_vocab._token2id.keys()

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
