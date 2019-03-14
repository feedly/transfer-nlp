# Build an experiment
To build a new experiment, you need:

- A `Vocabulary` (vocabularies implemented in `loaders/vocabulary.py` should be sufficient but you can extend to your own implemenations)
- A `Vectorizer` (some vectorizers are already implemented in `loaders/vectorizers.py`, but you can create your own by extending the `Vectorizer` class)
- A `CustomDataset` which will use your own dataset and create the vocabulary and vectorizer for you. A custom dataset class should extend the `CustomDataset` class
- A model: you can see some examples in `models/`, to create a custom model you need to extend the `nn.Module` class from Pytorch
- A `json` file containing all the experiments parameters (see `experiments/` for examples).
- Abstraction for the RunnerABC objects are defined in `runners/runnersABC.py` and `runners/instantiations.py`.
- In `runners/runner.py`, implement the `train_one_epoch` method, and run you experiment from you `json` exoeriment file. 

Important: an experiment `json` file must contain all necessary hyperparameters.
For the model, it should clearly contain the names of hyperparameters and names of inputs of the model.
For Optimizer, Scheduler, Loss function, Generator, Data and (evaluation) Metric.
See examples in the `/experiments/` folder.

Visualize training details using `tensorboard --logdir path/to/tensorboard/logs`
