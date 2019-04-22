import string
from collections import Counter
from typing import Dict, List, Any

from transfer_nlp.plugins.helpers import ObjectHyperParams
import numpy as np
import pandas as pd
import torch

from transfer_nlp.common.tokenizers import CustomTokenizer
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset, DatasetHyperParams
from transfer_nlp.loaders.vectorizers import VectorizerNew
from transfer_nlp.loaders.vocabulary import Vocabulary, SequenceVocabulary
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.predictors.predictor import Predictor, PredictorHyperParams


#Vectorizer class
@register_plugin
class NewsVectorizerNew(VectorizerNew):
    def __init__(self, data_file: str, cutoff: int):

        super().__init__(data_file=data_file)
        self.cutoff = cutoff

        self.tokenizer = CustomTokenizer()
        df = pd.read_csv(data_file)

        target_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(df.category)):
            target_vocab.add_token(category)

        word_counts = Counter()
        max_title = 0
        for title in df.title:
            tokens = self.tokenizer.tokenize(text=title)
            max_title = max(max_title, len(tokens))
            for token in tokens:
                if token not in string.punctuation:
                    word_counts[token] += 1

        data_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= self.cutoff:
                data_vocab.add_token(word)

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self.max_title = max_title + 2

    def vectorize(self, title: str) -> np.array:

        tokens = self.tokenizer.tokenize(text=title)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)
        vector_length = self.max_title
        # if vector_length < 0:
        #     vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector


# Dataset class
@register_plugin
class NewsDatasetSplits(DatasetSplits):

    def __init__(self, data_file:str, batch_size: int, dataset_hyper_params: DatasetHyperParams):

        self.df = pd.read_csv(data_file)

        # preprocessing
        self.vectorizer: VectorizerNew = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.title), axis=1)
        self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.category), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in','y_target']]
        val_df = self.df[self.df.split == 'val'][['x_in','y_target']]
        test_df = self.df[self.df.split == 'test'][['x_in','y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)

        # Class weights
        class_counts = self.df.category.value_counts().to_dict()
        sorted_counts = sorted(class_counts.items(), key=lambda x: self.vectorizer.target_vocab.lookup_token(x[0]))
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict:

        row = self._target_df.iloc[index]

        title_vector = self.vectorizer.vectorize(input_string=row.title)

        class_index = self.vectorizer.target_vocab.lookup_token(row.category)
        print(f"row: {row}")
        print(f"title vector: {title_vector}")
        print(f"class index: {class_index}")

        return {
            'x_in': title_vector,
            'y_target': class_index}


# Predictors
@register_plugin
class NewsPredictor(Predictor):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, predictor_hyper_params: PredictorHyperParams):
        super().__init__(predictor_hyper_params=predictor_hyper_params)

    def json_to_data(self, input_json: Dict) -> Dict:
        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(title=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)

        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]