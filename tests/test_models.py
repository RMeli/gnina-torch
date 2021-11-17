import pytest
import torch

from gnina.models import DenseBlock, models_dict


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def dims():
    return (12, 24, 24, 24)


@pytest.fixture
def x(batch_size, dims, device):
    return torch.normal(mean=0, std=1, size=(batch_size, *dims), device=device)


@pytest.mark.parametrize("model", ["default2017", "default2018", "dense"])
def test_forward_pose(batch_size, dims, x, device, model):
    """
    Test forward pass of models for pose prediction.
    """
    m = models_dict[(model, False)](input_dims=dims).to(device)
    pose_raw = m(x)

    assert pose_raw.shape == (batch_size, 2)


@pytest.mark.parametrize("model", ["default2017", "default2018", "dense"])
def test_forward_affinity(batch_size, dims, x, device, model):
    """
    Test forward pass of models for pose and binding affinity prediction.
    """
    m = models_dict[(model, True)](input_dims=dims).to(device)
    pose_log, affinity = m(x)

    assert pose_log.shape == (batch_size, 2)
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
    assert x.shape[0] == batch_size


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
    assert x.shape[0] == batch_size