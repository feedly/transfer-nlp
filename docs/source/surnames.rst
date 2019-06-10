Surnames Classification
=======================

A use case that arise very often in the book NLP with PyTorch is that of surnames classification: a dataset of names
from different countries is provided and the task is to predict the country.

Vectorizer
----------
The most straightforward to represent a surname is to get its one-hot character encoding:


.. code-block:: python

    import pandas as pd
    import numpy as np
    from transfer_nlp.loaders.vocabulary import Vocabulary


    @register_plugin
    class MyVectorizer(Vectorizer):

        def __init__(self, data_file: str):

            super().__init__(data_file=data_file)

            df = pd.read_csv(data_file)
            data_vocab = Vocabulary(unk_token='@')
            target_vocab = Vocabulary(add_unk=False)

            # Add surnames and nationalities to vocabulary
            for index, row in df.iterrows():
                surname = row.surname
                nationality = row.nationality
                data_vocab.add_many(tokens=surname)
                target_vocab.add_token(token=nationality)

            self.data_vocab = data_vocab
            self.target_vocab = target_vocab

        def vectorize(self, input_string: str) -> np.array:

            encoding = np.zeros(shape=len(self.data_vocab), dtype=np.float32)
            for character in surname:
                encoding[self.data_vocab.lookup_token(token=character)] = 1

            return encoding


Data loader
-----------
Let's create a data loader and have the PyTorch loaders set for train, vaildation and test categories.


.. code-block:: python


    from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset, DatasetHyperParams

    @register_plugin
    class MyDataLoader(DatasetSplits):

        def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
            self.df = pd.read_csv(data_file)
            self.vectorizer: Vectorizer = dataset_hyper_params.vectorizer

            self.df['x_in'] = self.df.apply(lambda row: self.vectorizer.vectorize(row.surname), axis=1)
            self.df['y_target'] = self.df.apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.nationality), axis=1)

            train_df = self.df[self.df.split == 'train'][['x_in', 'y_target']]
            val_df = self.df[self.df.split == 'val'][['x_in', 'y_target']]
            test_df = self.df[self.df.split == 'test'][['x_in', 'y_target']]

            super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                             val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                             test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


Model
-----

A simple modeling approach is to take the character one-hot encoding as input to a multi-layer perceptron:


.. code-block:: python

    import torch

    @register_plugin
    class ModelHyperParams(ObjectHyperParams):

        def __init__(self, dataset_splits: DatasetSplits):
            super().__init__()
            self.input_dim = len(dataset_splits.vectorizer.data_vocab)
            self.output_dim = len(dataset_splits.vectorizer.target_vocab)


    @register_plugin
    class MultiLayerPerceptron(torch.nn.Module):

        def __init__(self, model_hyper_params: ObjectHyperParams, hidden_dim: int):
            super(MultiLayerPerceptron, self).__init__()

            self.input_dim = model_hyper_params.input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = model_hyper_params.output_dim

            self.fc1 = torch.nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
            self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=self.output_dim)

        def forward(self, x_in: torch.tensor) -> torch.tensor:
            """
            Linear -> ReLu -> Linear (+ softmax if probabilities needed)
            :param x_in: size (batch, input_dim)
            :return:
            """
            intermediate = torch.nn.functional.relu(self.fc1(x_in))
            output = self.fc2(intermediate)

            if self.output_dim == 1:
                output = output.squeeze()

            return output


Predictor
---------

To use the model in inference mode, we create a specific predictor object:


.. code-block:: python

    from transfer_nlp.plugins.predictors import PredictorABC, PredictorHyperParams
    from transfer_nlp.plugins.config import register_plugin

    @register_plugin
    class MyPredictor(PredictorABC):

        def __init__(self, predictor_hyper_params: PredictorHyperParams):
            super().__init__(predictor_hyper_params=predictor_hyper_params)

        def json_to_data(self, input_json: Dict):
            return {
                'x_in': torch.tensor([self.vectorizer.vectorize(input_string=input_string) for input_string in input_json['inputs']])}

        def output_to_json(self, outputs: List) -> Dict[str, Any]:
            return {
                "outputs": outputs}

        def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probability_values, indices = probabilities.max(dim=1)
            return [{
                "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
                "probability": float(res[0])} for res in zip(probability_values, indices)]


Experiment
----------

Now that all classes are properly designed, we can define an experiment in a config file and have it trained:

.. code-block:: python

    from transfer_nlp.plugins.config import ExperimentConfig

    experiment_config = {
  "predictor": {
    "_name": "MLPPredictor",
    "data": "$my_dataset_splits",
    "model": "$model"
  },
  "my_dataset_splits": {
    "_name": "SurnamesDatasetMLP",
    "data_file": "$HOME/surnames/surnames_with_splits.csv",
    "batch_size": 128,
    "vectorizer": {
      "_name": "SurnamesVectorizerMLP",
      "data_file": "$HOME/surnames/surnames_with_splits.csv"
    }
  },
  "model": {
    "_name": "MultiLayerPerceptron",
    "hidden_dim": 100,
    "data": "$my_dataset_splits"
  },
  "optimizer": {
    "_name": "Adam",
    "lr": 0.01,
    "alpha": 0.99,
    "params": {
      "_name": "TrainableParameters"
    }
  },
  "scheduler": {
    "_name": "ReduceLROnPlateau",
    "patience": 1,
    "mode": "min",
    "factor": 0.5
  },
  "trainer": {
    "_name": "BasicTrainer",
    "model": "$model",
    "dataset_splits": "$my_dataset_splits",
    "loss": {
      "_name": "CrossEntropyLoss"
    },
    "optimizer": "$optimizer",
    "gradient_clipping": 0.25,
    "num_epochs": 5,
    "seed": 1337,
    "regularizer": {
      "_name": "L1"
    },
    "tensorboard_logs": "$HOME/surnames/tensorboard/mlp",
    "metrics": {
      "accuracy": {
        "_name": "Accuracy"
      },
      "loss": {
        "_name": "LossMetric",
        "loss_fn": {
          "_name": "CrossEntropyLoss"
        }
      }
    }
  }

    # Configure the experiment
    experiment = ExperimentConfig(experiment_config)
    # Launch the training loop
    experiment['trainer'].train()
    # Use the predictor for inference
    input_json = {"inputs": ["Zhang", "Mueller", "Rastapopoulos"]}
    output_json = experiment['predictor'].json_to_json(input_json=input_json)


