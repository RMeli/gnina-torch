import torch
import torch.nn as nn
from torch import Tensor


class PseudoHuberLoss(nn.Module):
    """
    GNINA pseudo-Huber loss.

    Notes
    -----
    Translated from the original custom Caffe layer. Not all functionality is
    implemented.

    https://github.com/gnina/gnina/blob/master/caffe/src/caffe/layers/affinity_loss_layer.cpp

    Definition of preudo-Huber loss:
    https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
    """

    def __init__(
        self, reduction: str = "mean", delta: float = 1.0, penalty: float = 0.0
    ):
        super().__init__()

        self.delta: float = delta
        self.delta2: float = delta * delta
        self.penalty: float = penalty

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

        # TODO: add diff_for_zero parameter

        scaled_diff = diff / self.delta

        loss = self.delta2 * (torch.sqrt(1.0 + scaled_diff * scaled_diff) - 1.0)

        if self.reduction == "sum":
            return torch.sum(loss)
        else:
            return torch.mean(loss)
