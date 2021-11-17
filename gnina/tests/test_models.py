import pytest
import torch

from gnina.models import Default2017, Default2018, Dense, DenseBlock


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def dims():
    return (12, 24, 24, 24)


@pytest.fixture
def x(batch_size, dims, device):
    return torch.normal(mean=0, std=1, size=(batch_size, *dims), device=device)


def test_default2017_forward(batch_size, dims, x, device):
    model = Default2017(input_dims=dims).to(device)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size,)


def test_default2018_forward(batch_size, dims, x, device):
    model = Default2018(input_dims=dims).to(device)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size,)


@pytest.mark.parametrize("num_block_convs", [1, 4])
@pytest.mark.parametrize("num_block_features", [2, 16])
def test_denseblock_forward_small(
    batch_size, x, num_block_features, num_block_convs, device
):
    in_features = x.shape[1]

    block = DenseBlock(
        in_features,
        num_block_features=num_block_features,
        num_block_convs=num_block_convs,
    ).to(device)
    x = block(x)

    # in_features from input
    # block_features for each of the convolutional layers
    assert x.shape[1] == num_block_features * num_block_convs + in_features
    assert x.shape[1] == block.out_features()


@pytest.mark.parametrize("num_block_convs", [1, 4])
@pytest.mark.parametrize("num_block_features", [8, 16])
def test_denseblock_forward(batch_size, x, num_block_features, num_block_convs, device):
    in_features = x.shape[1]

    block = DenseBlock(
        in_features,
        num_block_features=num_block_features,
        num_block_convs=num_block_convs,
    ).to(device)
    x = block(x)

    # in_features from input
    # block_features for each of the convolutional layers
    assert x.shape[1] == num_block_features * num_block_convs + in_features
    assert x.shape[1] == block.out_features()


@pytest.mark.parametrize("num_block_convs", [1, 4])
@pytest.mark.parametrize("num_block_features", [8, 16])
def test_dense_forward(
    batch_size, x, dims, num_block_features, num_block_convs, device
):
    model = Dense(
        input_dims=dims,
        num_blocks=3,
        num_block_features=num_block_features,
        num_block_convs=num_block_convs,
    ).to(device)
    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size,)
