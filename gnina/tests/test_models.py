import torch
import pytest

from gnina.models import Default2017, Default2018


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def dims():
    return (32, 48, 48, 48)


@pytest.fixture
def x(batch_size, dims):
    return torch.normal(mean=0, std=1, size=(batch_size, *dims))


def test_default2017_forward(batch_size, dims, x):
    model = Default2017(input_dims=dims)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size, 1)


def test_default2018_forward(batch_size, dims, x):
    model = Default2018(input_dims=dims)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size, 1)
