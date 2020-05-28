<img src="https://github.com/feedly/transfer-nlp/blob/v0.1/data/images/TransferNLP_Logo.jpg" width="1000">

Welcome to the Transfer NLP library, a framework built on top of PyTorch to promote reproducible experimentation and Transfer Learning in NLP

You can have an overview of the high-level API on this [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza), which shows how to use the framework on several examples.
All DL-based examples on these notebooks embed in-cell Tensorboard training monitoring!

For an example of pre-trained model finetuning, we provide a short executable tutorial on BertClassifier finetuning on this [Colab Notebook](https://colab.research.google.com/drive/10Toyi0V4fp0Sn33RSPCkoPrtf5FVpm3q#scrollTo=PXJFfulWkEl6)

# Set up your environment

```
mkvirtualenv transfernlp
workon transfernlp

git clone https://github.com/feedly/transfer-nlp.git
cd transfer-nlp
pip install -r requirements.txt
```

To use Transfer NLP as a library:

```
# to install the experiment builder only
pip install transfernlp
# to install Transfer NLP with PyTorch and Transfer Learning in NLP support
pip install transfernlp[torch]
```
or 
```
pip install git+https://github.com/feedly/transfer-nlp.git
```
to get the latest state before new releases.

To use Transfer NLP with associated examples:

```
git clone https://github.com/feedly/transfer-nlp.git
pip install -r requirements.txt
```

# Documentation
API documentation and an overview of the library can be found [here](https://transfer-nlp.readthedocs.io/en/latest/)

# Reproducible Experiment Manager
The core of the library is made of an experiment builder: you define the different objects that your experiment needs, and the configuration loader builds them in a nice way. For reproducible research and easy ablation studies, the library then enforces the use of configuration files for experiments.
As people have different tastes for what constitutes a good experiment file, the library allows for experiments defined in several formats:

- Python Dictionary
- JSON
- YAML
- TOML

In Transfer-NLP, an experiment config file contains all the necessary information to define entirely the experiment.
This is where you will insert names of the different components your experiment will use, along with the hyperparameters you want to use.
Transfer-NLP makes use of the Inversion of Control pattern, which allows you to define any class / method / function you could need, the `ExperimentConfig` class will create a dictionnary and instatiate your objects accordingly.

To use your own classes inside Transfer-NLP, you need to register them using the `@register_plugin` decorator. Instead of using a different registry for each kind of component (Models, Data loaders, Vectorizers, Optimizers, ...), only a single registry is used here, in order to enforce total customization.

If you use Transfer NLP as a dev dependency only, you might want to use it declaratively only, and call `register_plugin()` on objects you want to use at experiment running time. 

Here is an example of how you can define an experiment in a YAML file:

```
data_loader:
  _name: MyDataLoader
  data_parameter: foo
  data_vectorizer:
    _name: MyVectorizer
    vectorizer_parameter: bar

model:
  _name: MyModel
  model_hyper_param: 100
  data: $data_loader

trainer:
  _name: MyTrainer
  model: $model
  data: $data_loader
  loss:
    _name: PyTorchLoss
  tensorboard_logs: $HOME/path/to/tensorboard/logs
  metrics:
    accuracy:
      _name: Accuracy
```

Any object can be defined through a class, method or function, given a `_name` parameters followed by its own parameters.
Experiments are then loaded and instantiated using `ExperimentConfig(experiment=experiment_path_or_dict)`

Some considerations:

- Defaults parameters can be skipped in the experiment file.

- If an object is used in different places, you can refer to it using the `$` symbol, for example here the `trainer` object uses the `data_loader` instantiated elsewhere. No ordering of objects is required.

- For paths, you might want to use environment variables so that other machines can also run your experiments.
In the previous example, you would run e.g. `ExperimentConfig(experiment=yaml_path, HOME=Path.home())` to instantiate the experiment and replace `$HOME` by your machine home path.

- The config instantiation allows for any complex settings with nested dict / list

You can have a look at the [tests](https://github.com/feedly/transfer-nlp/blob/master/tests/plugins/test_config.py) for examples of experiment settings the config loader can build.
Additionally we provide runnable experiments in [`experiments/`](https://github.com/feedly/transfer-nlp/tree/master/experiments).

# Transfer Learning in NLP: flexible PyTorch Trainers
For deep learning experiments, we provide a `BaseIgniteTrainer` in [`transfer_nlp.plugins.trainers.py`](https://github.com/feedly/transfer-nlp/blob/master/transfer_nlp/plugins/trainers.py).
This basic trainer will take a model and some data as input, and run a whole training pipeline. We make use of the [PyTorch-Ignite](https://github.com/pytorch/ignite) library to monitor events during training (logging some metrics, manipulating learning rates, checkpointing models, etc...). Tensorboard logs are also included as an option, you will have to specify a `tensorboard_logs` simple parameters path in the config file. Then just run `tensorboard --logdir=path/to/logs` in a terminal and you can monitor your experiment while it's training!
Tensorboard comes with very nice utilities to keep track of the norms of your model weights, histograms, distributions, visualizing embeddings, etc so we really recommend using it.

<img src="https://github.com/feedly/transfer-nlp/blob/v0.1/data/images/tensorboard.png" width="1000">

We provide a `SingleTaskTrainer` class which you can use for any supervised setting dealing with one task.
We are working on a `MultiTaskTrainer` class to deal with multi task settings, and a `SingleTaskFineTuner` for large models finetuning settings.

# Use cases
Here are a few use cases for Transfer NLP:

- You have all your classes / methods / functions ready. Transfer NLP allows for a clean way to centralize loading and executing your experiments
- You have all your classes but you would like to benchmark multiple configuration settings: the `ExperimentRunner` class allows for sequentially running your sets of experiments, and generates personalized reporting (you only need to implement your `report` method in a custom `ReporterABC` class)
- You want to experiment with training deep learning models but you feel overwhelmed bby all the boilerplate code in SOTA models github projects. Transfer NLP encourages separation of important objects so that you can focus on the PyTorch `Module` implementation and let the trainers deal with the training part (while still controlling most of the training parameters through the experiment file)
- You want to experiment with more advanced training strategies, but you are more interested in the ideas than the implementations details. We are working on improving the advanced trainers so that it will be easier to try new ideas for multi task settings, fine-tuning strategies or model adaptation schemes. 


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
