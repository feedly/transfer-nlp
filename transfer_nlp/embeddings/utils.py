from typing import Dict, List, Tuple

import torch


def pretty_print(results: List[Tuple[str, torch.Tensor]]):
    """
    Pretty print embedding results.
    """
    for item in results:
        print("...[%.2f] - %s" % (item[1], item[0]))


def get_closest(target_word: str, word_to_idx: Dict, embeddings: torch.Tensor, n: int = 5) -> List[Tuple[str, torch.Tensor]]:
    """
    Get the n closest
    words to your word.
    """

    # Calculate distances to all other words

    word_embedding = embeddings[word_to_idx[target_word.lower()]]
    distances = []
    for word, index in word_to_idx.items():
        if word == "<MASK>" or word == target_word:
            continue
        distances.append((word, torch.dist(word_embedding, embeddings[index])))

    results = sorted(distances, key=lambda x: x[1])[1:n + 2]
    return results
