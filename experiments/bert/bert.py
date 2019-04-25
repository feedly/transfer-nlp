import math

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required
from tqdm import tqdm

from transfer_nlp.loaders.loaders import DatasetSplits, DatasetHyperParams, DataFrameDataset
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

    def vectorize(self, title: str, max_seq_length: int) -> np.array:
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

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)
        # np.random.shuffle(self.df.values)  # Use this code in dev mode
        # N = 100
        # self.df = self.df.head(n=N)

        # preprocessing
        self.vectorizer: Vectorizer = dataset_hyper_params.vectorizer

        self.max_sequence = 0
        for title in tqdm(self.df.title, desc="Getting max sequence"):
            tokens = self.vectorizer.tokenizer.tokenize(text=title)
            self.max_sequence = max(self.max_sequence, len(tokens))
        self.max_sequence += 2

        self.df['x_in'] = self.df['title'].progress_apply(lambda x: self.vectorizer.vectorize(title=x, max_seq_length=self.max_sequence))
        self.df['input_ids'] = self.df['x_in'].progress_apply(lambda x: x[0])
        self.df['attention_mask'] = self.df['x_in'].progress_apply(lambda x: x[1])
        self.df['token_type_ids'] = self.df['x_in'].progress_apply(lambda x: x[2])
        self.df['y_target'] = self.df['category'].progress_apply(lambda x: self.vectorizer.target_vocab.lookup_token(x))
        # self.df['labels'] = [None]*len(self.df)

        train_df = self.df[self.df.split == 'train'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]
        val_df = self.df[self.df.split == 'val'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]
        test_df = self.df[self.df.split == 'test'][['input_ids', 'attention_mask', 'token_type_ids', 'y_target']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


@register_plugin
def bert_model():
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)


# Optimizer Code from HuggingFace repo
def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


@register_plugin
class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
