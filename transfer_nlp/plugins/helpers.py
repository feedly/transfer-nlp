import torch.nn as nn

from transfer_nlp.plugins.config import register_plugin


@register_plugin
class ObjectHyperParams:
    """
    Use or extend this class to configure model hyper parameters that cannot be predetermined. E.g.
    a model input size that depends on the data set composition.
    """

    def __init__(self):
        self.input_dim: int = None
        self.output_dim: int = None


@register_plugin
class TrainableParameters:
    """
    Use this class to configure optimizer parameters.
    """

    def __init__(self, model):
        self.model: nn.Module = model

    def __iter__(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p
