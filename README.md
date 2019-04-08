# transfer-nlp
Welcome to the transfer-nlp library, a framework built on top of PyTorch whose goal is to progressively achieve 2 kinds of Transfer:

- easy trasfer of code --> the framework should be modular enough so that you don't have to re-write everything each time you experiment with a new architecture / a new kind of task
- easy transfer learning --> the framework should be able to easliy interact with pre-trained models and manipulate them in order to fine-tune some of their parts.

You can have an overview of the high-level API on this [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza), which shows how to use the framework on several examples.
All examples on these notebooks embed in-cell Tensorboard training monitoring!

Before starting using this repository:

- create a virtual environment: `mkvirtualenv YourEnvName`
- clone the repository: `git clone https://github.com/feedly/transfer-nlp.git`
- Install requirements: `pip install -r requirements.txt`

The library is available on [Pypi](https://pypi.org/project/transfer-nlp/) but ```pip install transfer-nlp``` is not recommended yet.

Structure of the library:

`loaders`
- `transfer-nlp/loaders/vocabulary.py`: contains classes for vocabularies
- `transfer-nlp/loaders/vectorizers.py`: classes for vectorizers
- `transfer-nlp/loaders/loaders.py`: classes for dataset loaders

`transfer-nlp/models/`: contains implementations of NLP models

`transfer-nlp/embeddings`: contains utility functions for embeddings management

`transfer-nlp/experiments`: each experiment is defined as a json config file, defining the whole experiment

`transfer-nlp/colab_experiments`: experiments for colab notebooks

`transfer-nlp/runners`: contains the full training pipeline, given a config file experiment


#How to use the library?

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

This is very useful if you want to set up a very custom training strategy, but for usual cases the plugins that are already implemented will be sufficient.


 Some objectves to reach:
 - Unit-test everything
 - Smooth the runner pipeline to enable multi-task training (without constraining the way we do multi-task, whether linear, hierarchical or else...)
 - Include examples using state of the art pre-trained models
 - Enable slack integration for model crashing / completion
 - Enable embeddings visualisation (see this project https://projector.tensorflow.org/)
 - Enable pre-trained models finetuning
 - Include linguistic properties to models
 - Experiment with RL for sequential tasks
 - Include probing tasks to try to understand the properties that are learned by the models



This library builds on the book <cite>["Natural Language Processing with PyTorch"](https://www.amazon.com/dp/1491978236/)<cite> by Delip Rao and Brian McMahan for the initial experiments.
