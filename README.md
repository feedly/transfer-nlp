# transfer-nlp
Welcome to the transfer-nlp library, a framework built on top of PyTorch whose goal is to progressively achieve 2 kinds of Transfer:

- easy trasfer of code --> the framework should be modular enough so that you don't have to re-write everything each time you experiment with a new architecture / a new kind of task
- easy transfer learning --> the framework should be able to easliy interact with pre-trained models and manipulate them in order to fine-tune some of their parts.

You can have an overview of the high-level API on this [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza), which shows how to use the framework on several examples.
All examples on these notebooks embed in-cell Tensorboard training monitoring!

# Set up your environment

```
mkvirtualenv transfernlp
workon transfernlp

git clone https://github.com/feedly/transfer-nlp.git
cd transfer-nlp
pip install -r requirements.txt
```

- create a virtual environment: `mkvirtualenv YourEnvName`
- clone the repository: `git clone https://github.com/feedly/transfer-nlp.git`
- Install requirements: `pip install -r requirements.txt`

The library is available on [Pypi](https://pypi.org/project/transfer-nlp/) but ```pip install transfer-nlp``` is not recommended yet.

# How to use the library?

You can have a look at the [Colab Notebook](https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=IuBcpSdZtcmo) to get a simple sense of the library usage.

A basic usage is:

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet.png)

You can use this code with all existing experiments built in the repo.

# How to define custom models?

Using your own models in the framework is made easy through the @register_model decorator. Here is an example:

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet2.png)

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

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet3.png)

Example of a metric:

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet4.png)

Example of a loss function:

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet5.png)

...and similarily for optimizers, schedulers, dataset loaders

This is very useful if you want to set up a very custom training strategy, but for usual cases the plugins that are already implemented will be sufficient.


# Tensorboard training monitoring
PyTorch comes with a handy TensorboardX integration that enable the use of Tensorboard.
Once the training process in launched you can run:

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet7.png)


# Slack integration
While experimenting with your own models / data, the training might take some time. To get notified when your training finishes or crashes, we recommend the simple library [knockknock](https://github.com/huggingface/knockknock) by folks at HuggingFace, which add a simple decorator to your running function to notify you via Slack, E-mail, etc.

![alt text](https://github.com/feedly/transfer-nlp/blob/master/data/snippets/snippet6.png)

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
