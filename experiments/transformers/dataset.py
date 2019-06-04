import pandas as pd
import torch
from pytorch_pretrained_bert import cached_path, BertTokenizer
import numpy as np
import random

from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset
from transfer_nlp.plugins.config import register_plugin


def load_data_lm():
    dataset_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/"
                               "wikitext-103-train-tokenized-bert.bin")
    datasets = torch.load(dataset_file)

    # Convert our encoded dataset to torch.tensors and reshape in blocks of the transformer's input length
    for split_name in ['train', 'valid']:
        tensor = torch.tensor(datasets[split_name], dtype=torch.long)
        num_sequences = (tensor.size(0) // 256) * 256
        datasets[split_name] = tensor.narrow(0, 0, num_sequences).view(-1, 256)

    n = len(datasets['valid']) // 2
    datasets['test'] = datasets['valid'][n:]
    datasets['valid'] = datasets['valid'][:n]
    datasets['train'] = datasets['train'][:1000]
    return datasets


def integerify(l):
    return [x.numpy() for x in l]


@register_plugin
class BertLMTuningDataset(DatasetSplits):

    def __init__(self, batch_size: int):
        datasets = load_data_lm()
        self.data = datasets

        train_df = pd.DataFrame(data={
            "x": integerify(datasets['train']),
            "y_target": integerify(datasets['train'])})
        val_df = pd.DataFrame(data={
            "x": integerify(datasets['valid']),
            "y_target": integerify(datasets['valid'])})
        test_df = pd.DataFrame(data={
            "x": integerify(datasets['test']),
            "y_target": integerify(datasets['test'])})

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size, val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
class BertCLFFinetuningDataset(DatasetSplits):

    def __init__(self, batch_size: int):
        dataset_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/trec/"
                                   "trec-tokenized-bert.bin")
        datasets = torch.load(dataset_file)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        for split_name in ['train', 'test']:
            # Trim the samples to the transformer's input length minus 1 & add a classification token
            datasets[split_name] = [x[:256 - 1] + [tokenizer.vocab['[CLS]']]
                                    for x in datasets[split_name]]

            # Pad the dataset to max length
            padding_length = max(len(x) for x in datasets[split_name])
            datasets[split_name] = [np.array(x + [tokenizer.vocab['[PAD]']] * (padding_length - len(x)))
                                    for x in datasets[split_name]]

            # # Convert to torch.Tensor and gather inputs and labels
            # tensor = torch.tensor(datasets[split_name], dtype=torch.long)
            # labels = torch.tensor(datasets[split_name + '_labels'], dtype=torch.long)
            # datasets[split_name] = TensorDataset(tensor, labels)

        valid_size = int(0.1 * len(datasets['train']))
        c = list(zip(datasets['train'], datasets['train_labels']))
        random.shuffle(c)
        datasets['train'], datasets['train_labels'] = zip(*c)
        datasets['train'], datasets['train_labels'] = list(datasets['train']), list(datasets['train_labels'])
        # np.random.shuffle(datasets['train'])

        datasets['valid'], datasets['valid_labels'] = datasets['train'][:valid_size], datasets['train_labels'][:valid_size]
        datasets['train'], datasets['train_labels'] = datasets['train'][valid_size:], datasets['train_labels'][valid_size:]


        train_df = pd.DataFrame(data={
            "x": datasets['train'],
            "y_target": datasets['train_labels'],
            "clf_tokens_mask": [np.array([int(x == tokenizer.vocab['[CLS]']) for x in example]) for example in datasets['train']]
        })
        val_df = pd.DataFrame(data={
            "x": datasets['valid'],
            "y_target": datasets['valid_labels'],
            "clf_tokens_mask": [np.array([int(x == tokenizer.vocab['[CLS]']) for x in example]) for example in datasets['valid']]
        })
        test_df = pd.DataFrame(data={
            "x": datasets['test'],
            "y_target": datasets['test_labels'],
            "clf_tokens_mask": [np.array([int(x == tokenizer.vocab['[CLS]']) for x in example]) for example in datasets['test']]
        })

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size, val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)
