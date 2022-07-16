import os

import mlflow

from gninatorch import inference, training


def test_inference(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1
    chekpointfile = os.path.join(str(tmpdir), f"checkpoint_{epochs}.pt")

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

    with mlflow.start_run():
        training.training(args_train)

        # Check that training was performed and produced a checkpoint file
        assert os.path.isfile(os.path.join(tmpdir, "training.log"))
        assert os.path.isfile(chekpointfile)

        inference.inference(args)

    # Confirm inference output files exist
    assert os.path.isfile(os.path.join(tmpdir, "inference.log"))
    assert os.path.isfile(os.path.join(tmpdir, "inference_results.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "inference_metrics.csv"))


def test_inference_affinity(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1
    chekpointfile = os.path.join(str(tmpdir), f"checkpoint_{epochs}.pt")
    model = "default2017"

    common_args = [
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
    ]

    # Use training function in order to create a checkpoint
    args_train = training.options(
        [
            trainfile,
            "-m",
            model,
            "--no_shuffle",
            "--test_every",
            "1",
            "--checkpoint_every",
            "1",
            "--iterations",
            str(epochs),
        ]
        + common_args
    )

    args = inference.options(
        [
            testfile,
            "default2017",
            chekpointfile,
        ]
        + common_args
    )

    with mlflow.start_run():
        training.training(args_train)

        # Check that training was performed and produced a checkpoint file
        assert os.path.isfile(os.path.join(tmpdir, "training.log"))
        assert os.path.isfile(chekpointfile)

        inference.inference(args)

    # Confirm inference output files exist
    assert os.path.isfile(os.path.join(tmpdir, "inference.log"))
    assert os.path.isfile(os.path.join(tmpdir, "inference_results.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "inference_metrics.csv"))


def test_inference_flex(trainfile, testfile, dataroot, tmpdir, device):
    epochs = 1
    chekpointfile = os.path.join(str(tmpdir), f"checkpoint_{epochs}.pt")
    model = "default2017"

    common_args = [
        "-d",
        dataroot,
        "--label_pos",
        "0",
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
    ]

    # Use training function in order to create a checkpoint
    args_train = training.options(
        [
            trainfile,
            "-m",
            model,
            "--no_shuffle",
            "--test_every",
            "1",
            "--checkpoint_every",
            "1",
            "--iterations",
            str(epochs),
        ]
        + common_args
    )

    args = inference.options(
        [
            testfile,
            model,
            chekpointfile,
        ]
        + common_args
    )

    with mlflow.start_run():
        training.training(args_train)

        # Check that training was performed and produced a checkpoint file
        assert os.path.isfile(os.path.join(tmpdir, "training.log"))
        assert os.path.isfile(chekpointfile)

        inference.inference(args)

    # Confirm inference output files exist
    assert os.path.isfile(os.path.join(tmpdir, "inference_results.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "inference_metrics.csv"))
