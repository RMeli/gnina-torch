import os

import pandas as pd
import pytest

from gnina import training


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

    # Check presence of output files
    assert os.path.isfile(os.path.join(tmpdir, "training.log"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_train.csv"))
    assert not os.path.isfile(os.path.join(tmpdir, "metrics_test.csv"))  # No test file

    df_train = pd.read_csv(os.path.join(tmpdir, "metrics_train.csv"))
    assert len(df_train) == 2


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

    # Check presence of output files
    assert os.path.isfile(os.path.join(tmpdir, "training.log"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_train.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_test.csv"))

    df_train = pd.read_csv(os.path.join(tmpdir, "metrics_train.csv"))
    df_test = pd.read_csv(os.path.join(tmpdir, "metrics_test.csv"))

    assert len(df_train) == 2
    assert len(df_test) == 2


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

    # Check presence of output files
    assert os.path.isfile(os.path.join(tmpdir, "training.log"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_train.csv"))
    assert not os.path.isfile(os.path.join(tmpdir, "metrics_test.csv"))

    df_train = pd.read_csv(os.path.join(tmpdir, "metrics_train.csv"))
    assert len(df_train) == 2


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
            "1",
            "--iterations",
            "5",
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
            "--progress_bar",
            "--base_lr",
            "0.1",
            "--lr_dynamic",
            "--lr_patience",
            "1",
            "--lr_min",
            "0.001",
        ]
    )

    training.training(args)

    # Check that the learning rate changes during training
    # TODO: Store learning rate internally and check the cache instead
    captured = capsys.readouterr()
    assert "Learning rate: 0.1" in captured.out  # Initial learning rate
    assert "Learning rate: 0.01" in captured.out  # Updated learning rate
    assert "Learning rate: 0.001" in captured.out  # Updated learning rate

    # Check presence of output files
    assert os.path.isfile(os.path.join(tmpdir, "training.log"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_train.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "metrics_test.csv"))

    df_train = pd.read_csv(os.path.join(tmpdir, "metrics_train.csv"))
    df_test = pd.read_csv(os.path.join(tmpdir, "metrics_test.csv"))

    assert len(df_train) == 5
    assert len(df_test) == 5
