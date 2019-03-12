# Build an experiment
To build a new experiment, you need:

- A `Vocabulary` (vocabularies implemented in `loaders/vocabulary.py` should be sufficient but you can extend to your own implemenations)
- A `Vectorizer` (some vectorizers are already implemented in `loaders/vectorizers.py`, but you can create your own by extending the `Vectorizer` class)
- A `CustomDataset` which will use your own dataset and create the vocabulary and vectorizer for you. A custom dataset class should extend the `CustomDataset` class
- A model: you can see some examples in `models/`, to create a custom model you need to extend the `nn.Module` class from Pytorch
- A `json` file containing all the experiments parameters (see `experiments/` for examples).
- In `runners/runner.py`, add you dataset and model classes, and run you experiment from you `json` exoeriment file. 

Visualize training details using `tensorboard --logdir path/to/tensorboard/logs`
