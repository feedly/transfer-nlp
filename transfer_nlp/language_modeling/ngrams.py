from collections import Counter
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


class NgramLM:

    def __init__(self, n: int):

        self.n = n  # Number of tokens in N-gram windows
        self.data: pd.DataFrame = None
        self.epsilon = 1e-8  # proba for N-grams that were not seen in the corpus
        self.buckets: Dict = None  # will contain counts for every n-grams, n from 1 to self.n
        self.cutoff = 10

    def load_data(self, data_path: Path):
        """

        :param data_path:
        :return:
        """

        self.data = pd.read_csv(filepath_or_buffer=data_path)

    def tokenize(self):
        """

        :return:
        """

        self.data['content'] = self.data['content'].apply(lambda x: str(x).lower())
        self.data['content'] = self.data['content'].apply(word_tokenize)

    def ngramize(self):
        """
        Compute the counts for each bucket of ngrams
        :return:
        """

        data = [item for sublist in self.data.content for item in sublist]
        buckets = {i: [' '.join(data[j:j+i]) for j in range(0, len(data), i)] for i in tqdm(range(1, self.n), f"N-gram bucket")}
        self.buckets = {key: Counter(value) for key, value in buckets.items()}
        self.buckets = {key: Counter(el for el in value.elements() if value[el] >= self.cutoff) for key, value in self.buckets.items()}
        # for key in self.buckets:
        #     self.buckets[i] =
        # self.buckets = {key: {word: value[word] for word in value if int(value[word]) > self.cutoff} for key, value in buckets.items()}

    def proba(self, n_plus_one: List[str], n: List[str]) -> float:
        """
        Compute the probability of next word given the past:
        P(next word | window) = P(window + word) / P(window)
        If window + word does not exist in corpus, we return self.epsilon
        If window does not exist in corpus, w decrease size of window until it exists
        :param n_plus_one:
        :param n:
        :return:
        """

        size = len(n_plus_one)
        n_plus_one = ' '.join(n_plus_one)
        if n_plus_one not in self.buckets[size]:
            # print(f"using default small value since {n_plus_one} is not in the vocabulary")
            return self.epsilon

        numerator = self.buckets[size][n_plus_one]

        size = len(n)
        while ' '.join(n) not in self.buckets[size] and len(n) > 1:
            size -= 1
            n = n[1:]

        n = ' '.join(n)
        denominator = self.buckets[size][n]

        return numerator / denominator

    def build_from_dataframe(self, data_path: Path):
        """
        Build the LM from a dataframe path
        :param data_path:
        :return:
        """

        self.load_data(data_path=data_path)
        self.tokenize()
        self.ngramize()

    def generate(self, first_token: str, number_token: int) -> str:
        """
        Generate text based on a first token
        :param first_token:
        :param number_token:
        :return:
        """

        if first_token not in self.buckets[1]:
            return f"{first_token} does not belong to our Vocabulary"

        result = [first_token]
        for i in range(number_token):
            probas = {word: self.proba(n_plus_one=result[-self.n + 2:] + [word], n=result[-self.n + 2:]) for word in self.buckets[1]}

            # Sample from the probability distribution over next word
            p = list(probas.values())
            p /= np.sum(p)
            next_word = np.random.choice(a=list(probas.keys()), size=1, p=p)[0]
            result.append(next_word)

        return ' '.join(result)


if __name__ == '__main__':

    save_path = Path.home() / 'work/experiments/nlp/data/feedly_data10000.csv'
    lm = NgramLM(n=5)
    lm.build_from_dataframe(data_path=save_path)
    lm.generate(first_token='the', number_token=20)
