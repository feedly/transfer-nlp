from pathlib import Path
from typing import Dict, List

import numpy as np
from annoy import AnnoyIndex
from smart_open import open
from tqdm import tqdm


class PreTrainedEmbeddings(object):

    def __init__(self, word_to_index: Dict, word_vectors: List[np.array]):

        self.w2i: Dict = word_to_index
        self.word_vectors: List[np.array] = word_vectors
        self.i2w: Dict = {v: k for k, v in self.w2i.items()}

        self.index: AnnoyIndex = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
        print("Building Index!")
        for _, i in tqdm(self.w2i.items(), 'Anoy Index'):
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)
        print("Finished!")

    @classmethod
    def from_embeddings_file(cls, embedding_file: Path):

        w2i = {}
        word_vectors = []

        with open(embedding_file) as fp:
            for line in tqdm(fp.readlines(), 'Embeddings'):
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])

                w2i[word] = len(w2i)
                word_vectors.append(vec)

        return cls(word_to_index=w2i, word_vectors=word_vectors)

    def get_embedding(self, word: str):

        return self.word_vectors[self.w2i[word]]

    def get_closest_to_vector(self, vector: np.array, n: int = 1):

        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.i2w[neighbor] for neighbor in nn_indices]

    def get_closest_to_word(self, word: str, n: int = 1):

        vector = self.get_embedding(word=word)
        return self.get_closest_to_vector(vector=vector, n=n)

    def compute_and_print_analogy(self, word1: str, word2: str, word3: str):

        vec1 = self.get_embedding(word=word1)
        vec2 = self.get_embedding(word=word2)
        vec3 = self.get_embedding(word=word3)

        spatial_relationship = vec2 - vec1
        vec4 = vec3 + spatial_relationship

        closest_words = self.get_closest_to_vector(vector=vec4, n=4)
        existing_words = set([word1, word2, word3])
        closest_words = [word for word in closest_words
                         if word not in existing_words]

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the computed vector!")
            return

        for word4 in closest_words:
            print("{} : {} :: {} : {}".format(word1, word2, word3, word4))


if __name__ == "__main__":
    embedding_file = Path.home() / 'work/experiments/nlp/data/glove/glove.6B.100d.txt'
    embeddings = PreTrainedEmbeddings.from_embeddings_file(embedding_file=embedding_file)

    print(embeddings.get_closest_to_word(word='house', n=10))
    embeddings.compute_and_print_analogy('man', 'he', 'woman')
