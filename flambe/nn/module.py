import math
from typing import Iterator, Tuple

import torch.nn as nn

from flambe.compile import Component


class Module(Component, nn.Module):
    """Base Flambé Module inteface.

    Provides the exact same interface as Pytorch's nn.Module, but
    extends it with a useful set of methods to access and clip
    parameters, as well as gradients.

    This abstraction allows users to convert their modules with a
    single line change, by importing from Flambé instead. Just like
    every Pytorch module, a forward method should be implemented.

    """

    @property
    def named_trainable_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """Get all the named parameters with `requires_grad=True`.

        Returns
        -------
        Iterator[Tuple[str, nn.Parameter]]
            Iterator over the parameters and their name.

        """
        parameters = filter(lambda p: p[1].requires_grad, self.named_parameters())
        return parameters

    @property
    def trainable_params(self) -> Iterator[nn.Parameter]:
        """Get all the parameters with `requires_grad=True`.

        Returns
        -------
        Iterator[nn.Parameter]
            Iterator over the parameters

        """
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return parameters

    @property
    def gradient_norm(self) -> float:
        """Compute the average gradient norm.

        Returns
        -------
        float
            The current average gradient norm

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad and p.grad is not None, self.parameters())
        norm = math.sqrt(sum(
            [param.grad.norm(p=2).item() ** 2 for param in parameters])  # type: ignore
        )
        return norm

    @property
    def parameter_norm(self) -> float:
        """Compute the average parameter norm.

        Returns
        -------
        float
            The current average parameter norm

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        norm = math.sqrt(
            sum([param.norm(p=2).item() ** 2 for param in parameters])  # type: ignore
        )
        return norm

    def num_parameters(self, trainable=False) -> int:
        """Gets the number of parameters in the model.

        Returns
        ----------
        int
            number of model params

        """
        # filter by trainable parameters
        if trainable:
            model_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        else:
            model_params = list(self.parameters())

        return(sum([len(x.view(-1)) for x in model_params]))  # type: ignore

    def clip_params(self, threshold: float):
        """Clip the parameters to the given range.

        Parameters
        ----------
        float
            Values are clipped between -threshold, threshold

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        for param in parameters:
            param.data.clamp_(min=-threshold, max=threshold)

    def clip_gradient_norm(self, threshold: float):
        """Clip the norm of the gradient by the given value.

        Parameters
        ----------
        float
            Threshold to clip at

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad and p.grad is not None, self.parameters())
        nn.utils.clip_grad_norm_(parameters, threshold)
