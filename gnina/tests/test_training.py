import os
import sys

import pytest
import torch

from gnina import training

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


def test_options_default(trainfile):
    args = training.options([trainfile])

    assert args.model == "default2017"
    assert args.data_root == ""
    assert args.gpu == "cuda:0"
    assert args.seed is None


@pytest.mark.parametrize("model", ["default2017", "default2018", "dense"])
@pytest.mark.parametrize("gpu", ["cpu", "cuda:1"])
def test_options(trainfile, model, gpu):
    seed = 42
    data_root = "data/"

    args = training.options(
        [trainfile, "-m", model, "-s", str(seed), "-d", data_root, "-g", gpu]
    )

    assert args.model == model
    assert args.data_root == data_root
    assert args.gpu == gpu
    assert args.seed == seed


def test_setup_example_provider_and_grid_maker_default(trainfile, dataroot):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    assert not args.shuffle

    e, gmaker, dims = training._setup_example_provider_and_grid_maker(args)

    assert e.num_labels() == 3  # Three labels in small.types
    assert e.size() == 2  # Two examples in small.types
    assert e.num_types() == 28
    assert gmaker.get_dimension() == pytest.approx(23.5)
    assert gmaker.get_resolution() == pytest.approx(0.5)
    assert gmaker.grid_dimensions(28) == (28, 48, 48, 48)
    assert dims == (28, 48, 48, 48)


def test_example_provider(trainfile, dataroot):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    assert not args.shuffle

    e, gmaker, dims = training._setup_example_provider_and_grid_maker(args)

    batch_size = 2

    batch = e.next_batch(batch_size)

    labels = torch.zeros(batch_size, device=device)
    affinities = torch.zeros(batch_size, device=device)
    whatever = torch.zeros(batch_size, device=device)

    assert args.label_pos == 0
    assert args.affinity_pos == 1
    batch.extract_label(args.label_pos, labels)
    batch.extract_label(args.affinity_pos, affinities)
    batch.extract_label(2, whatever)

    # Labels need to be transformed from float to long
    labels = labels.long()

    assert torch.allclose(labels, torch.tensor([0, 1], device=device))
    assert torch.allclose(affinities, torch.tensor([1.1, 2.1], device=device))
    assert torch.allclose(whatever, torch.tensor([1.2, 2.2], device=device))


def test_grid_maker(trainfile, dataroot):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options([trainfile, "-d", dataroot, "--no_shuffle"])

    assert not args.shuffle

    e, gmaker, dims = training._setup_example_provider_and_grid_maker(args)

    batch_size = 2

    batch = e.next_batch(batch_size)

    grid = torch.zeros((batch_size, *dims), device=device)
    assert not any(grid[grid > 0.0])

    gmaker.forward(batch, grid, random_translation=0, random_rotation=False)

    # Check that the grid has non-zero elements
    assert any(grid[grid > 0.0])


def test_training(trainfile):
    args = training.options([trainfile])
    training.training(args)
