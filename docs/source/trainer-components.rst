Trainer Components
===================

While the framework is flexible enough to deal with any kind of trainers, we encourage the use of a framework to manage
your training loops. We found that `Ignite <https://pytorch.org/ignite/>`_ provides everything we could expect from
a training management system.

Ignite defines 6 classes of events, defining a training loop:

- STARTED: start the training loop
- EPOCH_STARTED: start an epoch
- ITERATION_STARTED: start processing of one batch
- ITERATION_COMPLETED: complete processing of one batch
- EPOCH_COMPLETED: complete a full epoch
- COMPLETED: complete the training loop

Ignite allows to perform some actions at each of these events, by simply adding events.

Here are some examples of events you can do:

- Track metrics and log them on the terminal

- Log metrics, parameters norms, histograms, distributions, etc.. to Tensorboard (via TensorboardX)

- Learning schedulers: adapt the learning rates at different times of the training. A good example is the Cyclical learning rate scheduling, which has proven successful in models like `ULMFit <https://arxiv.org/abs/1801.06146>`_

- Model checkpointing: save your model periodically if it improves

- Early stopping: stop training when no learning is ever observed

- Terminate on NaNs: terminates the training when nans or infinite values are encountered.

- Timers

- ...

We provide a `BasicTrainer` class which should set you up for most cases in the supervised single task setting.
For more complex settings like multi-task learning, you might want to change the `_update` and `_inference` methods
to fit several tasks objectives / loss functions.


