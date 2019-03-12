from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


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


def make_embedding_matrix(glove_filepath: Path, words: List[str]) -> np.array:
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """

    w2i, glove_embeddings = load_glove_from_file(glove_filepath=glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in w2i:
            final_embeddings[i, :] = glove_embeddings[w2i[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings
