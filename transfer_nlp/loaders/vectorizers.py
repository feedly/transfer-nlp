import string
from collections import Counter
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

from transfer_nlp.common.tokenizers import CustomTokenizer
from transfer_nlp.loaders.vocabulary import Vocabulary, SequenceVocabulary
from transfer_nlp.plugins.config import register_plugin


class Vectorizer:

    def __init__(self, data_file: str):
        self.data_file = data_file
        # self.df = pd.read_csv(data_file)

    def vectorize(self, input_string: str):
        raise NotImplementedError
