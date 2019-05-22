import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from tqdm import tqdm

from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary
from transfer_nlp.plugins.config import register_plugin, ExperimentConfig

tqdm.pandas()


@register_plugin
class BertVectorizer(Vectorizer):
    def __init__(self, data_file: str, bert_version: str):
        super().__init__(data_file=data_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)
        df = pd.read_csv(data_file)
        self.target_vocab = Vocabulary(add_unk=False)
        self.target_vocab.add_many(set(df.category))

    def vectorize(self, title: str, max_seq_length: int) -> Tuple[np.array, np.array, np.array]:
        tokens = ["[CLS]"] + self.tokenizer.tokenize(title) + ["[SEP]"]
        token_type_ids, input_ids = [0] * len(tokens), self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask, padding = [1] * len(input_ids), [0] * (max_seq_length - len(input_ids))
        input_ids, attention_mask, token_type_ids = [x + padding for x in [input_ids, attention_mask, token_type_ids]]
        return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids)


@register_plugin
class BertDataloader(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, max_sequence: int, vectorizer: Vectorizer):
        self.vectorizer: Vectorizer = vectorizer
        self.max_sequence: int = max_sequence + 2
        df = pd.read_csv(data_file)
        df[['input_ids', 'attention_mask', 'token_type_ids']] = df.progress_apply(
            lambda row: pd.Series(self.vectorizer.vectorize(title=row['title'], max_seq_length=self.max_sequence)), axis=1)
        df['y_target'] = df['category'].progress_apply(lambda x: self.vectorizer.target_vocab.lookup_token(x))
        train_df, val_df, test_df = (df[df.split == mode][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']] for mode in ['train', 'val', 'test'])
        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size, val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
def bert_model(pretrained_model_name_or_path: str = 'bert-base-uncased', num_labels: int = 4):
    return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, num_labels=num_labels)


register_plugin(BertAdam)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    path = './bert.json'
    experiment = ExperimentConfig(path, HOME=home_env)
    experiment.experiment['trainer'].train()
