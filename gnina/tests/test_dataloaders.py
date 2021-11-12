import os
import sys

import pytest
import torch

from gnina import setup, training
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


@pytest.mark.parametrize("iteration_scheme", ["small", "large"])
def test_GriddedExamplesLoader(trainfile, dataroot, device, iteration_scheme):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "1",
            "--iteration_scheme",
            iteration_scheme,
        ]
    )

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert dataset.num_examples == 3
    assert dataset.num_labels == 3

    # Without balancing the length of the dataset is the same as the number of examples
    # This is true for both small and large epochs
    assert len(dataset) == 3

    for _ in range(len(dataset)):
        grids, labels = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)

    with pytest.raises(StopIteration):
        next(dataset)


def test_GriddedExamplesLoader_iteration_scheme_balanced_batch_size_1(
    trainfile, dataroot, device
):
    """
    Notes
    -----
    With :code:`batch_size` set to 1 as here, the sampling can't be balanced since the
    batch only contains a single example.
    """

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--balanced", "--batch_size", "1"]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        device=device,
    )

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
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
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert len(dataset_small) == 2  # Twice the minority class
    assert len(dataset_large) == 4  # Twice the majority class

    # With a batch_size of 1, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    grids, labels = next(dataset_small)
    assert grids.shape == (1, 28, 48, 48, 48)
    assert labels.shape == (1,)
    with pytest.raises(StopIteration):
        next(dataset_small)

    # With a batch_size of 1, three batches can be loaded in a large epoch since there
    # are two examples in the manjority class
    for _ in range(3):
        grids, labels = next(dataset_large)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)
    with pytest.raises(StopIteration):
        next(dataset_large)


def test_GriddedExamplesLoader_iteration_scheme_balanced_batch_size_2(
    trainfile, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--balanced", "--batch_size", "2"]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        device=device,
    )

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--iteration_scheme",
            "large",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert len(dataset_small) == 2  # Twice the minority class
    assert len(dataset_large) == 4  # Twice the majority class

    # With a batch_size of 2, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    grids, labels = next(dataset_small)
    assert grids.shape == (2, 28, 48, 48, 48)
    assert labels.shape == (2,)
    # The positive example is sampled once, with one of the negative examples
    assert torch.allclose(labels, torch.tensor([1, 0], device=device))
    with pytest.raises(StopIteration):
        next(dataset_small)

    # With a batch_size of 2, only two batches can be loaded in a large epoch since
    # there are two examples in the manjority class
    for _ in range(2):
        grids, labels = next(dataset_large)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)
        # The positive example is sampled twice, one for each of the negative examples
        assert torch.allclose(labels, torch.tensor([1, 0], device=device))
    with pytest.raises(StopIteration):
        next(dataset_large)


def test_GriddedExamplesLoader_batch_size(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--batch_size", "2"]
    )

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)

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
