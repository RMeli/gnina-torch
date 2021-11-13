import os
import sys

import pytest

from gnina import inference, training


@pytest.fixture
def trainfile() -> str:
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test.types")


@pytest.fixture
def testfile() -> str:
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test.types")


@pytest.fixture
def dataroot() -> str:
    gnina_path = os.path.dirname(sys.modules["gnina"].__file__)
    return os.path.join(gnina_path, "data", "test")


def test_inference(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1

    # Use training function in order to create a checkpoint
    args_train = training.options(
        [
            trainfile,
            "-m",
            "default2017",
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "1",
            "--test_every",
            "1",
            "--checkpoint_every",
            "1",
            "--iterations",
            str(epochs),
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
        ]
    )

    training.training(args_train)

    # Confirm that checkpoint file exists
    chekpointfile = os.path.join(str(tmpdir), f"checkpoint_{epochs}.pt")
    assert os.path.isfile(chekpointfile)

    args = inference.options(
        [
            testfile,
            "default2017",
            chekpointfile,
            "-d",
            dataroot,
            "--batch_size",
            "1",
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
            "--label_pos",
            "0",
        ]
    )

    inference.inference(args)


def test_inference_affinity(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1

    # Use training function in order to create a checkpoint
    args_train = training.options(
        [
            trainfile,
            "-m",
            "default2017",
            "-d",
            dataroot,
            "--affinity_pos",
            "1",
            "--no_shuffle",
            "--batch_size",
            "1",
            "--test_every",
            "1",
            "--checkpoint_every",
            "1",
            "--iterations",
            str(epochs),
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
        ]
    )

    training.training(args_train)

    # Confirm that checkpoint file exists
    chekpointfile = os.path.join(str(tmpdir), f"checkpoint_{epochs}.pt")
    assert os.path.isfile(chekpointfile)

    args = inference.options(
        [
            testfile,
            "default2017",
            chekpointfile,
            "-d",
            dataroot,
            "--affinity_pos",
            "1",
            "--batch_size",
            "1",
            "-o",
            str(tmpdir),
            "-g",
            str(device),
            "--seed",
            "42",
            "--label_pos",
            "0",
        ]
    )

    inference.inference(args)
