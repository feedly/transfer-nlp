from typing import Tuple

import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from tqdm import tqdm

from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary
from transfer_nlp.plugins.config import register_plugin

tqdm.pandas()


@register_plugin
class BertVectorizer(Vectorizer):
    def __init__(self, data_file: str):
        super().__init__(data_file=data_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        df = pd.read_csv(data_file)
        target_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(df.category)):
            target_vocab.add_token(category)
        self.target_vocab = target_vocab

    def vectorize(self, title: str, max_seq_length: int) -> Tuple[np.array, np.array, np.array]:
        tokens = self.tokenizer.tokenize(title)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_type_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids)


@register_plugin
class BertDataset(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, vectorizer: Vectorizer):
        self.df = pd.read_csv(data_file)
        np.random.shuffle(self.df.values)  # Use this code in dev mode
        N = 1000
        self.df = self.df.head(n=N)

        # preprocessing
        self.vectorizer: Vectorizer = vectorizer

        self.max_sequence = 0
        for title in tqdm(self.df.title, desc="Getting max sequence"):
            tokens = self.vectorizer.tokenizer.tokenize(text=title)
            self.max_sequence = max(self.max_sequence, len(tokens))
        self.max_sequence += 2

        vectors = self.df['title'].progress_apply(lambda x: self.vectorizer.vectorize(title=x, max_seq_length=self.max_sequence))
        self.df['input_ids'] = vectors.progress_apply(lambda x: x[0])
        self.df['attention_mask'] = vectors.progress_apply(lambda x: x[1])
        self.df['token_type_ids'] = vectors.progress_apply(lambda x: x[2])
        self.df['y_target'] = self.df['category'].progress_apply(lambda x: self.vectorizer.target_vocab.lookup_token(x))

        train_df = self.df[self.df.split == 'train'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
def bert_model(pretrained_model_name_or_path: str = 'bert-base-uncased', num_labels: int = 4):
    return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, num_labels=num_labels)


register_plugin(BertAdam)
