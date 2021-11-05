import os
import sys

import pytest
import torch

from gnina import training


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


def test_setup_example_provider_default(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "-g", str(device)]
    )

    assert not args.shuffle

    e = training._setup_example_provider(args.trainfile, args)

    assert e.num_labels() == 3  # Three labels in small.types
    assert e.size() == 3  # Three examples in small.types
    assert e.num_types() == 28


def test_setup_grid_maker_default(trainfile, dataroot, device):
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "-g", str(device)]
    )

    gmaker = training._setup_grid_maker(args)

    assert gmaker.get_dimension() == pytest.approx(23.5)
    assert gmaker.get_resolution() == pytest.approx(0.5)
    assert gmaker.grid_dimensions(28) == (28, 48, 48, 48)


def test_setup_grid_maker_dimensio_and_resolution(trainfile, dataroot, device):
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "-g",
            str(device),
            "--dimension",
            "10.0",
            "--resolution",
            "1.0",
        ]
    )

    gmaker = training._setup_grid_maker(args)

    assert gmaker.get_dimension() == pytest.approx(10.0)
    assert gmaker.get_resolution() == pytest.approx(1.0)
    assert gmaker.grid_dimensions(28) == (28, 11, 11, 11)


def test_example_provider(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    batch_size = 2
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--affinity_pos",
            "1",
            "-g",
            str(device),
            "--batch_size",
            str(batch_size),
        ]
    )

    assert not args.shuffle

    e = training._setup_example_provider(args.trainfile, args)

    batch_size = 2

    batch = next(e)

    assert len(batch) == batch_size

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


def test_grid_maker(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    # Manually set batch size
    batch_size = 2
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "-g",
            str(device),
            "--batch_size",
            str(batch_size),
        ]
    )

    assert not args.shuffle

    e = training._setup_example_provider(args.trainfile, args)
    gmaker = training._setup_grid_maker(args)
    dims = gmaker.grid_dimensions(e.num_types())

    batch = next(e)

    grid = torch.zeros((batch_size, *dims), device=device)
    assert not any(grid[grid > 0.0])

    gmaker.forward(batch, grid, random_translation=0, random_rotation=False)

    # Check that the grid has non-zero elements
    assert any(grid[grid > 0.0])


def test_training(trainfile, dataroot, tmpdir, device):
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
            "--test_every",
            "2",
            "--iterations",
            "5",
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
        ]
    )

    training.training(args)


def test_training_with_test(trainfile, dataroot, tmpdir, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "--testfile",
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "1",
            "--test_every",
            "2",
            "--iterations",
            "5",
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
            "--progress_bar",
        ]
    )

    training.training(args)
