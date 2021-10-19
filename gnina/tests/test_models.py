import torch

from gnina.models import Default2017


def test_default2017_forward():
    batch_size = 16
    dims = (32, 48, 48, 48)

    model = Default2017(input_dims=dims)

    x = torch.normal(mean=0, std=1, size=(batch_size, *dims))

    pose_raw, affinity = model(x)

    assert pose_raw.shape == (batch_size, 2)
    assert affinity.shape == (batch_size, 1)
