import os
import sys

import pytest
import torch

from gnina import training
from gnina.dataloaders import GriddedExamplesLoader


@pytest.fixture
def trainfile() -> str:
    """
    Path to small training file.
    """
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test.types")


@pytest.fixture
def dataroot() -> str:
    """
    Path to test directory.
    """
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test")


def test_GriddedExamplesLoader(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--batch_size", "1"]
    )

    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert len(dataset) == 3
    assert dataset.num_labels == 3

    for _ in range(len(dataset)):
        grids, labels = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)

    with pytest.raises(StopIteration):
        next(dataset)


def test_GriddedExamplesLoader_epoch_type(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--batch_size", "1"]
    )

    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)

    dataset_small = GriddedExamplesLoader(
        example_provider=e,
        grid_maker=gmaker,
        device=device,
    )

    dataset_large = GriddedExamplesLoader(
        example_provider=e,
        grid_maker=gmaker,
        device=device,
    )

    # If sampling is not balanced, the small and large epoch are the same
    assert len(dataset_small) == len(dataset_large) == 3


def test_GriddedExamplesLoader_iteration_scheme_balanced(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--balanced", "--batch_size", "1"]
    )
    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)

    dataset_small = GriddedExamplesLoader(
        example_provider=e,
        grid_maker=gmaker,
        device=device,
    )

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "1",
            "--iteration_scheme",
            "large",
        ]
    )
    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)

    dataset_large = GriddedExamplesLoader(
        example_provider=e,
        grid_maker=gmaker,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert len(dataset_small) == 2  # Twice the minority class
    assert len(dataset_large) == 4  # Twice the majority class


def test_GriddedExamplesLoader_batch_size(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--batch_size", "2"]
    )

    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert dataset.num_labels == 3

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
