import os

from gnina import inference, training


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


def test_inference_flex(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1

    # Use training function in order to create a checkpoint
    args_train = training.options(
        [
            trainfile,
            "-m",
            "default2017",
            "-d",
            dataroot,
            "--flexlabel_pos",
            "2",
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
            "--flexlabel_pos",
            "2",
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
