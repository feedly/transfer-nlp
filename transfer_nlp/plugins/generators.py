"""
This file shows an example of batch generators implementations.
To implement your own batch generators, you should use the decorator @register_batch_generator which allows the framework to
reuse your custom batch generators
"""


from typing import Dict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transfer_nlp.plugins.registry import register_batch_generator


@register_batch_generator
def generate_batches(dataset: Dataset, batch_size: int, shuffle: bool=True, drop_last: bool=True, device: str='cpu') -> Dict:

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


@register_batch_generator
def generate_nmt_batches(dataset: Dataset, batch_size: int, shuffle: bool=True,
                         drop_last: bool=True, device: str="cpu") -> Dict:
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_lengths'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict