import numpy as np
import pytest
import torch
from torch import nn

from gninatorch import losses


def test_affinity_loss_perfect_predictions(device):
    criterion = losses.AffinityLoss()

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    assert criterion(target, target).item() == pytest.approx(0.0)


def test_affinity_loss_zero_affinity(device):
    criterion = losses.AffinityLoss()

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Predictions where target is zero are ignored
    predicted = torch.tensor([1.0, 1.1, 2.2, 3.3, 4.4], device=device)

    assert criterion(predicted, target).item() == pytest.approx(0.0)


def test_affinity_loss_wrong_predicted_affinity_good_pose_sum(device):
    criterion = losses.AffinityLoss(reduction="sum", pseudo_huber=True, scale=100.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with good pose
    predicted = torch.tensor([0.0, 1.1 + diff, 2.2, 3.3, 4.4], device=device)

    assert criterion(predicted, target).item() == pytest.approx(
        100 * (np.sqrt(1.0 + diff**2) - 1.0)
    )


def test_affinity_loss_wrong_predicted_affinity_good_pose_mean_PH(device):
    criterion = losses.AffinityLoss(reduction="mean", pseudo_huber=True, scale=100.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with good pose
    predicted = torch.tensor([0.0, 1.1 + diff, 2.2, 3.3, 4.4], device=device)

    # Divide by the lenght of the target to get mean loss
    expected_loss = 100 * (np.sqrt(1.0 + diff**2) - 1.0) / len(target)

    assert criterion(predicted, target).item() == pytest.approx(expected_loss)


def test_affinity_loss_overestimated_predicted_affinity_bad_pose_sum_PH(device):
    criterion = losses.AffinityLoss(reduction="sum", pseudo_huber=True, scale=100.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with bad pose
    # Affinity is overestimated
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 + diff, 4.4], device=device)

    expected_loss = 100 * (np.sqrt(1.0 + diff**2) - 1.0)

    assert criterion(predicted, target).item() == pytest.approx(expected_loss)


def test_affinity_loss_underestimated_predicted_affinity_bad_pose_sum_PH(device):
    criterion = losses.AffinityLoss(reduction="sum", pseudo_huber=True, scale=10.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with bad pose
    # Affinity is underestimated, which is what we want
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 - diff, 4.4], device=device)

    # Underestimated affinity for a bad pose does not contribute to the loss
    assert criterion(predicted, target).item() == pytest.approx(0.0)


def test_affinity_loss_wrong_predicted_affinity_good_pose_mean_L2(device):
    # Delta parameter only affects pseudo-Huber loss
    criterion = losses.AffinityLoss(reduction="mean", pseudo_huber=False, delta=4.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with good pose
    predicted = torch.tensor([0.0, 1.1 + diff, 2.2, 3.3, 4.4], device=device)

    assert criterion(predicted, target).item() == pytest.approx(diff**2 / len(target))


def test_affinity_loss_overestimated_predicted_affinity_bad_pose_sum_L2(device):
    # Delta parameter only affects pseudo-Huber loss
    criterion = losses.AffinityLoss(reduction="sum", pseudo_huber=False, delta=4.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with bad pose
    # Affinity is overestimated
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 + diff, 4.4], device=device)

    assert criterion(predicted, target).item() == pytest.approx(diff**2)


def test_affinity_loss_underestimated_predicted_affinity_bad_pose_sum_L2(device):
    # Delta parameter only affects pseudo-Huber loss
    criterion = losses.AffinityLoss(reduction="sum", pseudo_huber=False, delta=4.0)

    diff = 1.5

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4], device=device)

    # Affinity error with bad pose
    # Affinity is underestimated, which is what we want
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 - diff, 4.4], device=device)

    # Underestimated affinity for a bad pose does not contribute to the loss
    assert criterion(predicted, target).item() == pytest.approx(0.0)


def test_nllloss_scaled_null(device):
    criterion = losses.ScaledNLLLoss()

    tt = torch.tensor([0, 0, 1, 1, 0], device=device)
    it = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.1, 0.0], [0.0, 1.0]], device=device
    )
    assert criterion(it, tt).item() == pytest.approx(0.0)


def test_nllloss_scaled(device):
    scale = 0.5

    loss = losses.ScaledNLLLoss(scale=0.5)
    unscaled_loss = nn.NLLLoss()

    tt = torch.tensor([0, 0, 1, 1, 0], device=device)
    it = torch.tensor(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.2, 0.8], [0.2, 0.8]], device=device
    )

    assert loss(it, tt).item() == pytest.approx(scale * unscaled_loss(it, tt).item())
