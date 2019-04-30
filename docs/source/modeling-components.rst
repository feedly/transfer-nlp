Modeling Components
===================

While the framework is flexible enough to deal with any kind of objects, here are some baseline components that you
can use:


Models
------
A model extends the PyTorch `torch.nn.Module` class. You only have to define implement the `__init__` and the `forward`
classes. Your model class will have hyperparameters (which are used at object creation), and parameters for the
`forward` method (used when `__call__` is called). The parameters that the `forward` method expects should match the
parameters yield by the PyTorch batch iterator. For example:

.. code-block:: python

    import torch
    from transfer_nlp.plugins.config import register_plugin

    @register_plugin
    class MyClassifier(torch.nn.Module):
        def __init__(self, input_dim: int, ouput_dim: int):

            super(MyClassifier, self).__init__()

        def forward(self, input_tensor: torch.tensor):
            # Do complex transofmrations
            return result

In this example, you need to set your data loader to yield batches with the key `"input_tensor"`.
If the `forward` method has default parameters that do not appear in the batch, they will be used, otherwise tyey will
be replaced by the values from the batch

Optimizers
----------
Optimizers allows for moving the model parameters in the direction of their gradients, following the strategy proper of
a certain optimizer.
The framework registry comes with all PyTorch optimizers so you should be good to go for most cases, e.g.:

.. code-block:: python

    experiment_config = {
                           "optimizer": {"_name": "Adam",
                                        "params": "model_params"
                                        }
                        }


However, if you want to use a custom Optimizer, you need to extend the `torch.optim.Optimizer` class and
register it to the registry. For example, if we want to use the optimizer used for BERT, we can use this
`implementation <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/optimization.py>`_
and register it like this:


.. code-block:: python

    @register_plugin
    class BertAdam(Optimizer):

        def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                     b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                     max_grad_norm=1.0):

            super(BertAdam, self).__init__(params, defaults)
        def step(self, closure=None):
            # Compute the loss
            return loss


    experiment_config = {
                           "optimizer": {"_name": "BertAdam",
                                        "params": "model_params"
                                        }
                        }

