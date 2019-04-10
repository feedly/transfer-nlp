import torch

from transfer_nlp.plugins.registry import register_regularizer


class Regularizor:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def compute_penalty_uniform(self, model: torch.nn.modules):
        """
        Compute a penalty value uniformly over layers
        :param self:
        :param model:
        :return:
        """

        penalty = 0

        for name, parameter in model.named_parameters():
            penalty += self(parameter)

        return penalty


@register_regularizer
class L1(Regularizor):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


@register_regularizer
class L2(Regularizor):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))
