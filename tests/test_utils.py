import pytest
import torch

from gninatorch import utils


def test_set_device_cpu():
    device = utils.set_device("cpu")
    assert device.type == "cpu"
    assert device.index is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_set_device_gpu():
    device = utils.set_device("cuda")
    assert device.type == "cuda"
    assert device.index is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_set_device_gpu_index():
    device = utils.set_device("cuda:0")
    assert device.type == "cuda"
    assert device.index == 0
