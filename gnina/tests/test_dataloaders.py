import os
import sys

import pytest
import torch

from gnina import training
from gnina.dataloaders import GriddedExamplesLoader

# TODO: Allow to deactivate cuda when running tests
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def trainfile() -> str:
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test.types")


@pytest.fixture
def dataroot() -> str:
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test")


def test_GriddedExamplesLoader(trainfile, dataroot):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    e, gmaker = training._setup_example_provider_and_grid_maker(args)

    dataset = GriddedExamplesLoader(batch_size=1, example_provider=e, grid_maker=gmaker)

    assert len(dataset) == 3
    assert dataset.num_labels == 3
    assert dataset.num_batches == 3

    for _ in range(len(dataset)):
        grids, labels = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)

    with pytest.raises(StopIteration):
        next(dataset)


def test_GriddedExamplesLoader_batch_size(trainfile, dataroot):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    e, gmaker = training._setup_example_provider_and_grid_maker(args)

    dataset = GriddedExamplesLoader(
        batch_size=2, example_provider=e, grid_maker=gmaker, device=device
    )

    assert len(dataset) == 3
    assert dataset.num_labels == 3
    assert dataset.num_batches == 2

    grids, labels = next(dataset)
    assert grids.shape == (2, 28, 48, 48, 48)
    assert labels.shape == (2,)
    assert torch.allclose(labels, torch.tensor([0, 1], device=device))

    grids, labels = next(dataset)
    assert grids.shape == (2, 28, 48, 48, 48)
    assert labels.shape == (2,)
    # Last batch padded with examples from the next epoch
    #   https://gnina.github.io/libmolgrid/python/index.html#molgrid.ExampleProviderSettings.iteration_scheme
    assert torch.allclose(labels, torch.tensor([0, 0], device=device))


def test_GriddedExamplesLoader_oversized_batch(trainfile, dataroot):
    """
    Test batch bigger than the number of examples
    """
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    e, gmaker = training._setup_example_provider_and_grid_maker(args)

    with pytest.raises(ValueError):
        GriddedExamplesLoader(batch_size=5, example_provider=e, grid_maker=gmaker)
