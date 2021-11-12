import os
import sys

import pytest

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


def test_training_pose_and_affinity(trainfile, dataroot, tmpdir, device):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "2",
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
            "--label_pos",
            "0",
            "--affinity_pos",
            "1",
        ]
    )

    training.training(args)


def test_training_lr_scheduler(trainfile, dataroot, tmpdir, device, capsys):
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
            "--lr_dynamic",
            "--lr_patience",
            "1",
        ]
    )

    training.training(args)

    # Check that the learning rate changes during training
    # TODO: Store learning rate internally and check the cache instead
    captured = capsys.readouterr()
    assert "Learning rate: 0.01" in captured.out  # Original (default) learning rate
    assert "Learning rate: 0.001" in captured.out  # Updated learning rate
