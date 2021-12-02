import pytest
import torch

from gnina import transforms


@pytest.fixture
def batch_size():
    return 64


@pytest.fixture
def pose_log(batch_size, device):
    return torch.normal(mean=0, std=1, size=(batch_size, 2), device=device)


@pytest.fixture
def labels(batch_size, device):
    return torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)


@pytest.fixture
def affinity_pred(batch_size, device):
    return torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)


@pytest.fixture
def affinity_exp(batch_size, device):
    return torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)


@pytest.fixture
def output(pose_log, labels, affinity_pred, affinity_exp):
    return {
        "pose_log": pose_log,
        "labels": labels,
        "affinities_pred": affinity_pred,
        "affinities": affinity_exp,
    }


def test_output_transform_select_log_pose(pose_log, labels, output):
    pl, ll = transforms.output_transform_select_log_pose(output)

    assert torch.allclose(pl, pose_log)
    assert torch.allclose(ll, labels)


def test_output_transform_select_pose(pose_log, labels, output):
    p, ll = transforms.output_transform_select_pose(output)

    pose = torch.exp(pose_log)

    assert torch.allclose(p, pose)
    assert torch.allclose(ll, labels)


def test_output_transform_affinity(affinity_pred, affinity_exp, output):
    ap, ae = transforms.output_transform_select_affinity(output)

    assert torch.allclose(ap, affinity_pred)
    assert torch.allclose(ae, affinity_exp)


def test_output_transform_affinity_abs(affinity_pred, affinity_exp, output):
    # The experimental (target) affinity is returned as absolute value
    # This takes care of the negative sign used to denote affinities of bad poses
    ap, ae = transforms.output_transform_select_affinity_abs(output)

    assert torch.allclose(ap, affinity_pred)
    assert torch.allclose(ae, torch.abs(affinity_exp))


def test_output_transform_ROC(pose_log, labels, output):
    pp, ll = transforms.output_transform_ROC(output)

    # ROC calculations requite only the probability of the positive class
    pose_positive_class = torch.exp(pose_log)[:, -1]

    assert torch.allclose(pp, pose_positive_class)
    assert torch.allclose(ll, labels)
