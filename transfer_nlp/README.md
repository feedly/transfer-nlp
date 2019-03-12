# transfer-nlp
Building on D. Rao's book "Natural Language Processing with PyTorch", this library aims at bringing a flexible framework for NLP projects.

The ideal use of this library is to provide a minimal implementation of a dataset loader, a vectorizer and a model. Then, given a config file with the experiment parameters, `runner.py` takes care of the training pipeline.


Before starting using this repository:

- create a virtual environment: `mkvirtualenv YourEnvName`
- clone the repository: `git clone https://github.com/petermartigny/transfer-nlp.git`
- Install requirements: `pip install -r requirements.txt`

Structure of the library:

`loaders`
- `loaders/vocabulary.py`: contains classes for vocabularies
- `loaders/vectorizers.py`: classes for vectorizers
- `loaders/loaders.py`: classes for dataset loaders

`models`: contains implementations of NLP models

`embeddings`: contains utility functions for embeddings management

`experiments`: each experiment is defined as a json config file, defining the whole experiment

`runners`: contains the full training pipeline, given a config file experiment
