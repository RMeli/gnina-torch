import numpy as np
import pytest
import torch

from gnina import losses


def test_affinity_loss_perfect_predictions(device):
    criterion = losses.AffinityLoss()

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    assert criterion(target, target).item() == pytest.approx(0.0)


def test_affinity_loss_zero_affinity(device):
    criterion = losses.AffinityLoss()

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    # Predictions where target is zero are ignored
    predicted = torch.tensor([1.0, 1.1, 2.2, 3.3, 4.4])

    assert criterion(predicted, target).item() == pytest.approx(0.0)


def test_affinity_loss_wrong_predicted_affinity_good_pose_sum(device):
    criterion = losses.AffinityLoss(reduction="sum")

    diff = 0.1

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    # Affinity error with good pose
    predicted = torch.tensor([0.0, 1.1 + diff, 2.2, 3.3, 4.4])

    assert criterion(predicted, target).item() == pytest.approx(
        np.sqrt(1.0 + diff ** 2) - 1.0, abs=1e-6
    )


def test_affinity_loss_wrong_predicted_affinity_good_pose_mean(device):
    criterion = losses.AffinityLoss(reduction="mean")

    diff = 0.1

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    # Affinity error with good pose
    predicted = torch.tensor([0.0, 1.1 + diff, 2.2, 3.3, 4.4])

    # Divide by the lenght of the target to get mean loss
    expected_loss = (np.sqrt(1.0 + diff ** 2) - 1.0) / len(target)

    assert criterion(predicted, target).item() == pytest.approx(expected_loss, abs=1e-6)


def test_affinity_loss_overestimated_predicted_affinity_bad_pose_sum(device):
    criterion = losses.AffinityLoss(reduction="sum")

    diff = 0.1

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    # Affinity error with bad pose
    # Affinity is overestimated
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 + diff, 4.4])

    assert criterion(predicted, target).item() == pytest.approx(
        np.sqrt(1.0 + diff ** 2) - 1.0, abs=1e-6
    )


def test_affinity_loss_underestimated_predicted_affinity_bad_pose_sum(device):
    criterion = losses.AffinityLoss(reduction="sum")

    diff = 0.1

    target = torch.tensor([0.0, 1.1, 2.2, -3.3, -4.4])

    # Affinity error with bad pose
    # Affinity is underestimated, which is what we want
    predicted = torch.tensor([0.0, 1.1, 2.2, 3.3 - diff, 4.4])

    # Underestimated affinity for a bad pose does not contribute to the loss
    # TODO: should contribute with add diff_for_zero parameter
    assert criterion(predicted, target).item() == pytest.approx(0.0)
