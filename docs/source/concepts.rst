Concepts
========

Experiment
----------

The **essence** of the framework is the class :class:`~transfer_nlp.plugins.config.ExperimentConfig`, a class which
enables to define an experiment based on a json file. An experiment will contain all the components that you might need:
- Data loader
- Model
- Optimizer
- Trainer
- ...

Launching experiments from json config files has two main advantages:

- **reproducibility**: when you are happy with the outcome of an experiment, the json file you used defines it entirely, so it is really easy to reproduce
- **ablation studies**: when experimenting with new architectures, it is becoming a standards practice to assess the importance of some model components to the outcome.
Using json files facilitates this process, where you just have to remove some components from the json file and run the experiment again.


.. code-block:: python

    from transfer_nlp.plugins.config import ExperimentConfig

    # Defining an experiment and starting the training pipeline
    experiment_config = {...}  # Config dictionary with components defining your experiment
    experiment = ExperimentConfig(experiment_config)
    experiment['trainer'].train()

    # Using the trained model to make predictions on some inputs
    predictor = experiment['predictor']
    json_input = {'inputs': []}
    results = predictor.json_to_json(input_json=input_json)



Json file
---------

The class :class:`~transfer_nlp.plugins.config.ExperimentConfig` has been designed so that an experiment can be
instantiated from any kind of objects you might need.
The experiment instantiator is able to deal with 3 kinds of inputs from the json files:

- **Simple parameters**: these are simple user-defined values, such as:

.. code-block:: python

    experiment_config = {"lr": 0.01,
                       "seed": 1,
                       "num_epochs": 1}

- **simple lists**: this is the same as simple parameters, but using lists, e.g.:

.. code-block:: python

    experiment_config = {"layer_sizes": [10, 50, 10]}


- **complex configuration**: here you can instantiate an object from any class. The framework will require the json file
to contain the name of the used class, e.g.:

.. code-block:: python

    experiment_config = {"lr": 0.01,
                       "model": {"_name": "MyClassifier"}}

When creating an instance of the class, `ExperimentConfig` will check for the hyperparameters. If it does not find them
and the class defines default parameters, those will be used. Otherwise, an exception will be thrown. So in this example
if the `MyClassifier` class takes `input_dim` and `output_dim` as hyperparameters, you would define the experiment as:

.. code-block:: python

    experiment_config = {"lr": 0.01,
                       "input_dim": 10000,
                       "output_dim": 5,
                       "model": {"_name": "MyClassifier"}}

To let Transfer NLP know about your custom classes, you add them to a registry. The framework does not require using
separate registries for some fixed set of components, such as Models, Optimizers, etc..
There is an only one registry of classes, where you need to add your custom classes to use the framework.

Let's say you have a fancy model class that extends the PyTorch neural network module class. The only thing
you need to do is add the class to the registry using the `@register_plugin` decorator:


.. code-block:: python

    import torch
    from transfer_nlp.plugins.config import register_plugin

    @register_plugin
    class MyClassifier(torch.nn.Module):
        def __init__(self, input_dim: int, ouput_dim: int):

            super(MyClassifier, self).__init__()

        def forward(self, input_tensor):
            # Do complex transofmrations
            return result
