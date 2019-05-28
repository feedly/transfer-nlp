<img src="https://github.com/feedly/transfer-nlp/blob/v0.1/data/images/TransferNLP_Logo.jpg" width="1000">

Welcome to the Transfer NLP library, a framework built on top of PyTorch whose goal is to progressively achieve 2 kinds of Transfer:

- **easy transfer of code**: the framework should be modular enough so that you don't have to re-write everything each time you experiment with a new architecture / a new kind of task
- **easy transfer learning**: the framework should be able to easily interact with pre-trained models and manipulate them in order to fine-tune some of their parts.

You can have an overview of the high-level API on this [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza), which shows how to use the framework on several examples.
All examples on these notebooks embed in-cell Tensorboard training monitoring!

For an example of pre-trained model finetuning, we provide a short executable tutorial on BertClassifier finetuning on this [Colab Notebook](https://colab.research.google.com/drive/10Toyi0V4fp0Sn33RSPCkoPrtf5FVpm3q#scrollTo=PXJFfulWkEl6)

# Set up your environment

```
mkvirtualenv transfernlp
workon transfernlp

git clone https://github.com/feedly/transfer-nlp.git
cd transfer-nlp
pip install -r requirements.txt
```

- create a virtual environment: `mkvirtualenv YourEnvName` (with mkvirtualenv or your choice of virtual env manager)
- clone the repository: `git clone https://github.com/feedly/transfer-nlp.git`
- Install requirements: `pip install -r requirements.txt`

The library is available on [Pypi](https://pypi.org/project/transfer-nlp/) but ```pip install transfer-nlp``` is not recommended yet.

# Documentation
API documentation and an overview of the library can be found [here](https://transfer-nlp.readthedocs.io/en/latest/)

# High-Level usage of the library

You can have a look at the [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=IuBcpSdZtcmo) to get a simple sense of the library usage.

The core of the library is made of an experiment builder: you define the different objects that your experiment needs, and the configuration loader builds them in a nice way:

```
from transfer_nlp.plugins.config import ExperimentConfig

# Launch an experiment
config_file  = {...}  # Dictionary config file, or str/Path to a json config file
experiment = ExperimentConfig(experiment=config_file)

# Interaact with the experiment's objects, e.g. launch a training job of a `trainer` object
experiment['trainer'].train()

# Another use of experiment object: use the `predictor` object for inference
input_json = {"inputs": [Some Examples]}
output_json = experiment['predictor'].json_to_json(input_json=input_json)
```

# How to experiment with the library?
For reproducible research and easy ablation studies, the library enforces the use of configuration files for experiments.

In Transfer-NLP, an experiment config file contains all the necessary information to define entirely the experiment.
This is where you will insert names of the different components your experiment will use.
Transfer-NLP makes use of the Inversion of Control pattern, which allows you to define any kind of classes you could need, and the `ExperimentConfig.from_json` method will create a dictionnary and instatiate your objects accordingly.

To use your own classes inside Transfer-NLP, you need to register them using the `@register_plugin` decorator. Instead of using a different registry for each kind of component (Models, Data loaders, Vectorizers, Optimizers, ...), only a single registry is used here, in order to enforce total customization.

Currently, the config file logic has 3 kinds of components:

- simple parameters: those are parameters which you know the value in advance: 
```
{"initial_learning_rate": 0.01,
"embedding_dim": 100,...}
```
- simple lists: similar to simple parameters, but as a list:
```
{"layers_dropout": [0.1, 0.2, 0.3], ...}
```
- Complex config: this is where the library instantiates your objects: every object needs to have its `_name` specified (the name of a class that you have register through the `@register_plugin` decorator), and its parameters defined. If your class has default parameters and your config file doesn't contain them, objects will be instantiated as default. Otherwise the parameters have to be present in the config file. Sometimes, initialization parameters are not available before launching the experiment. E.g., suppose your Model object needs a Vocabulary size as init input. The config file logic here makes it easy to deal with this while keeping the library code very general. 

You can have a look at the [tests](https://github.com/feedly/transfer-nlp/blob/master/tests/plugins/test_config.py) for examples of experiment settings the config loader can build.
Additionally we provide runnable experiments in [`experiments/`](https://github.com/feedly/transfer-nlp/tree/master/experiments).

# Usage Pipeline
The goal of the config file is to load different objects and run your experiment from it. 

A very common object to use is trainer, which you will run during your experiment. We provide a `BasicTrainer` in [`transfer_nlp.plugins.trainers.py`](https://github.com/feedly/transfer-nlp/blob/master/transfer_nlp/plugins/trainers.py).
This basic trainer will take a model and some data as input, and run a whole training pipeline. We make use of the [PyTorch-Ignite](https://github.com/pytorch/ignite) library to monitor events during training (logging some metrics, manipulating learning rates, checkpointing models, etc...). Tensorboard logs are also included as an option, you will have to specify a `tensorboard_logs` simple parameters path in the config file. Then just run `tensorboard --logdir=path/to/logs` in a terminal and you can monitor your experiment while it's training!
Tensorboard comes with very nice utilities to keep track of the norms of your model weights, histograms, distributions, visualizing embeddings, etc so we really recommend using it.

<img src="https://github.com/feedly/transfer-nlp/blob/v0.1/data/images/tensorboard.png" width="1000">

# Slack integration
While experimenting with your own models / data, the training might take some time. To get notified when your training finishes or crashes, you can use the simple library [knockknock](https://github.com/huggingface/knockknock) by folks at HuggingFace, which add a simple decorator to your running function to notify you via Slack, E-mail, etc.


# Some objectives to reach:
 - Include examples using state of the art pre-trained models
 - Include linguistic properties to models
 - Experiment with RL for sequential tasks
 - Include probing tasks to try to understand the properties that are learned by the models

# Acknowledgment
The library has been inspired by the reading of <cite>["Natural Language Processing with PyTorch"](https://www.amazon.com/dp/1491978236/)<cite> by Delip Rao and Brian McMahan.
Experiments in [`experiments`](https://github.com/feedly/transfer-nlp/tree/master/experiments/deep_learning_with_pytorch), the Vocabulary building block and embeddings nearest neighbors are taken or adapted from the code provided in the book.
