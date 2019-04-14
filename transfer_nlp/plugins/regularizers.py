import torch

from transfer_nlp.plugins.registry import register_regularizer


class RegularizerABC:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        return self.__repr__()

    def compute_penalty(self, model: torch.nn.Module):
        raise NotImplementedError


@register_regularizer
class L1(RegularizerABC):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))

    def __str__(self):
        return f"L1(alpha={self.alpha})"

    def compute_penalty(self, model: torch.nn.Module):
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
class L2(RegularizerABC):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __str__(self):
        return f"L2(alpha={self.alpha})"

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))

    def compute_penalty(self, model: torch.nn.Module):
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