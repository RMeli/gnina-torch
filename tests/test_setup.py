import pytest
import torch

from gninatorch import setup, training


def test_setup_example_provider_default(trainfile, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "-g", str(device)]
    )

    assert not args.shuffle

    e = setup.setup_example_provider(args.trainfile, args)

    assert e.num_labels() == 3  # Three labels in small.types
    assert e.size() == 3  # Three examples in small.types
    assert e.num_types() == 28


def test_setup_grid_maker_default(trainfile, dataroot, device):
    args = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "-g", str(device)]
    )

    gmaker = setup.setup_grid_maker(args)

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

    gmaker = setup.setup_grid_maker(args)

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
            "--flexlabel_pos",
            "2",
            "-g",
            str(device),
            "--batch_size",
            str(batch_size),
        ]
    )

    assert not args.shuffle

    e = setup.setup_example_provider(args.trainfile, args)

    batch_size = 2

    batch = next(e)

    assert len(batch) == batch_size

    labels = torch.zeros(batch_size, device=device)
    affinities = torch.zeros(batch_size, device=device)
    flexlabels = torch.zeros(batch_size, device=device)

    assert args.label_pos == 0
    assert args.affinity_pos == 1
    batch.extract_label(args.label_pos, labels)
    batch.extract_label(args.affinity_pos, affinities)
    batch.extract_label(args.flexlabel_pos, flexlabels)

    # Labels need to be transformed from float to long
    labels = labels.long()
    flexlabels = flexlabels.long()

    assert torch.allclose(labels, torch.tensor([0, 1], device=device))
    assert torch.allclose(affinities, torch.tensor([1.1, 2.1], device=device))
    assert torch.allclose(flexlabels, torch.tensor([1, 1], device=device))


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

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)
    dims = gmaker.grid_dimensions(e.num_types())

    batch = next(e)

    grid = torch.zeros((batch_size, *dims), device=device)
    assert not any(grid[grid > 0.0])

    gmaker.forward(batch, grid, random_translation=0, random_rotation=False)

    # Check that the grid has non-zero elements
    assert any(grid[grid > 0.0])


def test_setup_example_provider_double_balance(trainfilestrat, dataroot, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfilestrat,
            "-d",
            dataroot,
            "--no_shuffle",
            "-g",
            str(device),
            "--balanced",
            "--stratify_pos",
            "2",
            "--stratify_min",
            "0",
            "--stratify_max",
            "1",
            "--stratify_step",
            "0.5",
        ]
    )

    assert not args.shuffle

    e = setup.setup_example_provider(args.trainfile, args)

    assert e.num_labels() == 3  # Three labels in small.types
    assert e.size() == 12  # Examples in small.types
    assert e.num_types() == 28
