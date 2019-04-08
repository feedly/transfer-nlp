# transfer-nlp
Welcome to the transfer-nlp library, a framework built on top of PyTorch whose goal is to progressively achieve 2 kinds of Transfer:

- easy trasfer of code --> the framework should be modular enough so that you don't have to re-write everything each time you experiment with a new architecture / a new kind of task
- easy transfer learning --> the framework should be able to easliy interact with pre-trained models and manipulate them in order to fine-tune some of their parts.

You can have an overview of the high-level API on this [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza), which shows how to use the framework on several examples.
All examples on these notebooks embed in-cell Tensorboard training monitoring!

# Set up your environment

- create a virtual environment: `mkvirtualenv YourEnvName`
- clone the repository: `git clone https://github.com/feedly/transfer-nlp.git`
- Install requirements: `pip install -r requirements.txt`

The library is available on [Pypi](https://pypi.org/project/transfer-nlp/) but ```pip install transfer-nlp``` is not recommended yet.

# Structure of the library:

`loaders`
- `transfer-nlp/loaders/vocabulary.py`: contains classes for vocabularies
- `transfer-nlp/loaders/vectorizers.py`: classes for vectorizers
- `transfer-nlp/loaders/loaders.py`: classes for dataset loaders

`transfer-nlp/models/`: contains implementations of NLP models

`transfer-nlp/embeddings`: contains utility functions for embeddings management

`transfer-nlp/experiments`: each experiment is defined as a json config file, defining the whole experiment

`transfer-nlp/colab_experiments`: experiments for colab notebooks

`transfer-nlp/runners`: contains the full training pipeline, given a config file experiment


# How to use the library?

You can have a look at the [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=IuBcpSdZtcmo) to get a simple sens of the library usage.

A basic usage is:

```
from transfer_nlp.runners.runner import Runner

config_args = {} # dictionary containing all parameters necessary to define an experiment. See /experiments/ for examples
runner = Runner(config_args=config_args)
runner.run(test_at_the_end=False)
```

You can use this code with all existing experiments built in the repo.

# How to define custom models?

Using your own models in the framework is made easy through the @register_model decorator. Here is an examples:

```
from transfer_nlp.plugins.registry import register_model
import torch.nn as nn
import torch
import torch.nn.functional as F

#Define your custom model
@register_model
class MyCustomMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MyCustomMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        """
        Linear -> ReLu -> Linear -> ReLu -> Linear (+ softmax if probabilities needed)
        :param x_in: size (batch, input_dim)
        :param apply_softmax: False if used with the cross entropy loss, True if probability wanted
        :return:
        """

        intermediate = F.relu(self.fc1(x_in))
        intermediate = F.relu(self.fc2(intermediate))
        output = self.fc3(intermediate)

        if self.output_dim == 1:
            output = output.squeeze()

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output
        
config_args = {"model": {
    "modelName": "MyCustomMLP",
    "modelParams": [
      "input_dim",
      "hidden_dim",
      "output_dim"
    ],
    "modelInputs": [
      "x_in"
    ]
  },
  ...}
runner = Runner(config_args=config_args)
runner.run(test_at_the_end=False)
```

# Customization

Using decorators @register_{thing to register}, it is possible to customize these components:

- Optimizers
- Schedulers
- Metrics
- Datasets loaders
- Models
- Loss functions
- Batch generators

Example of a batch generator:

```
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
        
config_args = {"batch_generator": "generate_batches",
  ...}
runner = Runner(config_args=config_args)
runner.run(test_at_the_end=False)
```

Example of a metric:

```
from transfer_nlp.plugins.registry import register_metric

@register_metric
def compute_accuracy_sequence(input, target, mask_index):
    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100
    
config_args = {"metrics": ["compute_accuracy_sequence"],
  ...}
runner = Runner(config_args=config_args)
runner.run(test_at_the_end=False)
```

Example of a loss function:

```
import torch.nn.functional as F

from transfer_nlp.plugins.metrics import normalize_sizes
from transfer_nlp.plugins.registry import register_loss


def sequence_loss(input, target, mask_index):

    y_pred, y_true = normalize_sizes(y_pred=input, y_true=target)
    return F.cross_entropy(input=y_pred, target=y_true, ignore_index=mask_index)


@register_loss
class SequenceLoss:

    def __init__(self):
        self.mask: bool = True

    def __call__(self, *args, **kwargs):
        return sequence_loss(*args, **kwargs)
    
config_args = {"loss": {
    "lossName": "SequenceLoss",
    "lossParams": []
  },
  ...}
runner = Runner(config_args=config_args)
runner.run(test_at_the_end=False)
```
...and similarily for optimizers, schedulers, dataset loaders


This is very useful if you want to set up a very custom training strategy, but for usual cases the plugins that are already implemented will be sufficient.


# Tensorboard training monitoring
PyTorch comes with a handy TensorboardX integration that enable the use of Tensorboard.
Once the training process in launched you can run:

```

```

# Slack integration
While experimenting with your own models / data, the training might take some time. To get notified when your training finishes or crashes, we recommend the simple library [knockknock](https://github.com/huggingface/knockknock) by folks at HuggingFace, which add a simple decorator to your running function to notify you via Slack, E-mail, etc.


```
from transfer_nlp.runners.runner import Runner

slack_webhook_url = "YourWebhookURL"
slack_channel = "YourFavoriteSlackChannel"
config_args = {}
runner = Runner(config_args=config_args)

@slack_sender(webhook_url=slack_webhook_url, channel=slack_channel)
def run_with_slack(runner):
    runner.run()
```


# Some objectves to reach:
 - Unit-test everything
 - Smooth the runner pipeline to enable multi-task training (without constraining the way we do multi-task, whether linear, hierarchical or else...)
 - Include examples using state of the art pre-trained models
 - Enable embeddings visualisation (see this project https://projector.tensorflow.org/)
 - Enable pre-trained models finetuning
 - Include linguistic properties to models
 - Experiment with RL for sequential tasks
 - Include probing tasks to try to understand the properties that are learned by the models



# Acknowledgment
This library builds on the book <cite>["Natural Language Processing with PyTorch"](https://www.amazon.com/dp/1491978236/)<cite> by Delip Rao and Brian McMahan for the initial experiments.
