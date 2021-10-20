import pytest
import torch

from gnina.models import Default2017, Default2018, DenseBlock

# TODO: Allow to deactivate cuda when running tests
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def dims():
    return (32, 48, 48, 48)


@pytest.fixture
def x(batch_size, dims):
    return torch.normal(mean=0, std=1, size=(batch_size, *dims), device=device)


def test_default2017_forward(batch_size, dims, x):
    model = Default2017(input_dims=dims).to(device)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size, 1)


def test_default2018_forward(batch_size, dims, x):
    model = Default2018(input_dims=dims).to(device)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size, 1)


@pytest.mark.parametrize("num_convs", [1, 4])
@pytest.mark.parametrize("block_features", [8, 16])
def test_denseblock_forward(batch_size, x, block_features, num_convs):
    in_features = x.shape[1]

    block = DenseBlock(
        in_features, block_features=block_features, num_convs=num_convs
    ).to(device)
    x = block(x)

    # in_features from input
    # block_features for each of the convolutional layers
    assert x.shape[1] == block_features * num_convs + in_features
