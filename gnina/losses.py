import torch
import torch.nn as nn
from torch import Tensor


class ScaledNLLLoss(nn.Module):
    """
    Scaled NLLLoss.

    Parameters
    ----------
    scale: float
        Scaling factor for the loss
    reduction: str
        Reduction method (mean or sum)
    """

    def __init__(
        self,
        scale: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.scale = scale
        self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Predicted values
        target: Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss
        """
        return self.scale * self.loss(input, target)


class AffinityLoss(nn.Module):
    """
    GNINA affinity loss.

    Parameters
    ----------
    reduction: str
        Reduction method (mean or sum)
    delta: float
        Scaling factor
    penalty: float
        Penalty factor
    pseudo_huber: bool
        Use pseudo-huber loss as opposed to L2 loss
    scale: float
        Scaling factor for the loss

    Notes
    -----
    Translated from the original custom Caffe layer. Not all functionality is
    implemented.

    https://github.com/gnina/gnina/blob/master/caffe/src/caffe/layers/affinity_loss_layer.cpp

    The :code:`scale` parameter is different from the original implementation. In the
    original Caffe implementation, the :code:`scale` parameter is used to scale the
    gradients in the backward pass. Here the scale parameter scales the loss function
    directly in the forward pass.

    Definition of pseudo-Huber loss:
    https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
    """

    def __init__(
        self,
        reduction: str = "mean",
        delta: float = 1.0,
        penalty: float = 0.0,
        pseudo_huber: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        self.delta: float = delta
        self.delta2: float = delta * delta
        self.penalty: float = penalty
        self.pseudo_huber: bool = pseudo_huber
        self.scale: float = scale

        assert reduction in ["mean", "sum"]
        self.reduction: str = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Predicted values
        target: Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss

        Notes
        -----
        Binding affinity (pK) is positive for good poses and negative for bad poses (and
        zero if unknown). This allows to distinguish good poses from bad poses (to
        which a penalty is applied) without explicitly using the labels or the RMSD.
        """
        assert input.size() == target.size()

        # Normal euclidean distance for good poses (positive affinity label)
        diff = torch.where(target > 0, input - target, torch.zeros_like(input))

        # Hinge-like distance for bad poses (negative affinity label)
        diff = torch.where(
            torch.logical_and(target < 0, target > -input),
            input + target + self.penalty,
            diff,
        )

        if self.pseudo_huber:
            scaled_diff = diff / self.delta
            loss = self.delta2 * (torch.sqrt(1.0 + scaled_diff * scaled_diff) - 1.0)
        else:  # L2 loss
            loss = diff * diff

        if self.reduction == "mean":
            reduced_loss = torch.mean(loss)
        else:  # Assertion in init ensures that reduction is "sum"
            reduced_loss = torch.sum(loss)

        return self.scale * reduced_loss
