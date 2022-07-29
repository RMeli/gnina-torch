import re

import molgrid
import numpy as np
import pytest
import torch

import gninatorch
from gninatorch import gnina, models
from gninatorch.dataloaders import GriddedExamplesLoader


@pytest.mark.parametrize(
    "input_key, output_key",
    [
        ("conv1.weight", "features.conv1.weight"),
        ("output_fc.weight", "pose.pose_output.weight"),
        ("output_fc_aff.weight", "affinity.affinity_output.weight"),
        ("pose_output.weight", "pose.pose_output.weight"),
        ("affinity_output.weight", "affinity.affinity_output.weight"),
    ],
)
def test_rename(input_key, output_key):
    assert gnina._rename(input_key) == output_key


def test_rename_wrong_name():
    with pytest.raises(RuntimeError, match="Unknown layer name"):
        gnina._rename("wrong_layer_name")


def test_load_gnina_model_default2017():
    model = gnina.load_gnina_model("default2017")

    assert isinstance(model, models.Default2017Affinity)
    assert model.input_dims == (35, 48, 48, 48)


@pytest.mark.parametrize(
    "model_name",
    ["crossdock_default2018"]
    + [f"crossdock_default2018_{i}" for i in range(1, 5)]
    + ["general_default2018"]
    + [f"general_default2018_{i}" for i in range(1, 5)]
    + ["redock_default2018"]
    + [f"redock_default2018_{i}" for i in range(1, 5)],
)
def test_load_gnina_model_default2018(model_name: str):
    model = gnina.load_gnina_model(model_name)

    assert isinstance(model, models.Default2018Affinity)
    assert model.input_dims == (28, 48, 48, 48)


@pytest.mark.parametrize("model_name", ["dense"] + [f"dense_{i}" for i in range(1, 5)])
def test_load_gnina_model_dense(model_name: str):
    model = gnina.load_gnina_model(model_name)

    assert isinstance(model, models.DenseAffinity)
    assert model.input_dims == (28, 48, 48, 48)


def test_load_gnina_model_wrong():
    with pytest.raises(ValueError, match="Unknown model name"):
        gnina.load_gnina_model("wrong_model_name")


@pytest.mark.parametrize(
    "model_name, CNNscore, CNNaffinity",
    [
        (
            "redock_default2018",
            np.array([0.02956, 0.00114, 0.00095]),
            np.array([1.31840, 1.20986, 1.14063]),
        ),
        (
            "general_default2018",
            np.array([0.32619, 0.37634, 0.39832]),
            np.array([1.28267, 1.36640, 1.50419]),
        ),
        (
            "crossdock_default2018",
            np.array([0.64764, 0.43467, 0.19287]),
            np.array([1.28360, 1.27934, 1.06574]),
        ),
        (
            "dense",
            np.array([0.94850, 0.82229, 0.65933]),
            np.array([1.93134, 1.81497, 1.56016]),
        ),
    ],
)
def test_gnina_model_prediction(
    dataroot, testfile, device, model_name, CNNscore, CNNaffinity
):
    """
    Test predictions of pre-trained models against GNINA predictions.

    Notes
    -----
    GNINA has been running as follows in order to generate the baseline:

    gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l1.sdf --score_only --cnn MODEL
    gnina -r tests/data/mols/r2.pdb -l tests/data/mols/l2.sdf --score_only --cnn MODEL
    gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l2.sdf --score_only --cnn MODEL
    """
    model = gnina.load_gnina_model(model_name)
    model.to(device)
    model.eval()

    ep = molgrid.ExampleProvider(
        data_root=dataroot,
        balanced=False,
        shuffle=False,
        default_batch_size=3,
        iteration_scheme=molgrid.IterationScheme.SmallEpoch,
    )
    ep.populate(testfile)

    gmaker = molgrid.GridMaker(resolution=0.5, dimension=23.5)

    dataset = GriddedExamplesLoader(
        example_provider=ep, grid_maker=gmaker, device=device
    )

    grids, _ = next(dataset)

    with torch.no_grad():
        log_pose, affinity = model(grids)

    assert log_pose.shape == (3, 2)
    assert affinity.shape == (3,)

    # Check that scores sum to one
    assert torch.allclose(
        torch.exp(log_pose).sum(dim=-1), torch.ones_like(affinity), atol=1e-5
    )

    score = torch.exp(log_pose)[:, -1].cpu().numpy()

    assert np.allclose(1 - score, 1 - CNNscore, atol=1e-5)
    assert np.allclose(affinity.cpu().numpy(), CNNaffinity, atol=1e-5)


@pytest.mark.parametrize(
    "model_name, CNNscore, CNNaffinity, CNNvariance",
    [
        (
            "redock_default2018",
            np.array([0.08090, 0.00871, 0.01234]),
            np.array([1.14039, 0.94944, 0.90828]),
            np.array([0.04115, 0.09732, 0.07626]),
        ),
        (
            "general_default2018",
            np.array([0.48171, 0.46914, 0.55628]),
            np.array([1.54386, 1.50739, 1.70184]),
            np.array([0.03959, 0.02752, 0.03837]),
        ),
        (
            "crossdock_default2018",
            np.array([0.60276, 0.29299, 0.22103]),
            np.array([1.15954, 1.07318, 0.94330]),
            np.array([0.09806, 0.09105, 0.09067]),
        ),
        (
            "dense",
            np.array([0.98567, 0.62727, 0.85364]),
            np.array([2.62781, 1.96368, 2.26030]),
            np.array([0.21371, 0.46775, 0.19200]),
        ),
    ],
)
def test_gnina_model_prediction_ensemble(
    dataroot, testfile, device, model_name, CNNscore, CNNaffinity, CNNvariance
):
    """
    Test predictions of pre-trained models against GNINA predictions.

    Notes
    -----
    GNINA has been running as follows in order to generate the baseline:

    gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l1.sdf --score_only --cnn MODEL_ensemble
    gnina -r tests/data/mols/r2.pdb -l tests/data/mols/l2.sdf --score_only --cnn MODEL_ensemble
    gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l2.sdf --score_only --cnn MODEL_ensemble
    """
    model = gnina.load_gnina_models(
        [model_name] + [f"{model_name}_{i}" for i in range(1, 5)]
    )
    model.to(device)
    model.eval()

    ep = molgrid.ExampleProvider(
        data_root=dataroot,
        balanced=False,
        shuffle=False,
        default_batch_size=3,
        iteration_scheme=molgrid.IterationScheme.SmallEpoch,
    )
    ep.populate(testfile)

    gmaker = molgrid.GridMaker(resolution=0.5, dimension=23.5)

    dataset = GriddedExamplesLoader(
        example_provider=ep, grid_maker=gmaker, device=device
    )

    grids, _ = next(dataset)

    with torch.no_grad():
        log_pose, affinity, affinity_var = model(grids)

    assert log_pose.shape == (3, 2)
    assert affinity.shape == (3,)
    assert affinity_var.shape == (3,)

    assert torch.allclose(
        torch.exp(log_pose).sum(dim=-1), torch.ones_like(affinity), atol=1e-5
    )

    score = torch.exp(log_pose)[:, -1].cpu().numpy()

    assert np.allclose(1 - score, 1 - CNNscore, atol=1e-5)
    assert np.allclose(affinity.cpu().numpy(), CNNaffinity, atol=1e-5)

    # Compare 1-affinity_var because variance is expected to be small
    assert np.allclose(1 - affinity_var.cpu().numpy(), 1 - CNNvariance, atol=1e-5)


@pytest.mark.parametrize(
    "model_name, CNNscore, CNNaffinity",
    [
        (
            "redock_default2018",
            np.array([0.02956, 0.00114, 0.00095]),
            np.array([1.31840, 1.20986, 1.14063]),
        ),
        (
            "general_default2018",
            np.array([0.32619, 0.37634, 0.39832]),
            np.array([1.28267, 1.36640, 1.50419]),
        ),
        (
            "crossdock_default2018",
            np.array([0.64764, 0.43467, 0.19287]),
            np.array([1.28360, 1.27934, 1.06574]),
        ),
        (
            "dense",
            np.array([0.94850, 0.82229, 0.65933]),
            np.array([1.93134, 1.81497, 1.56016]),
        ),
    ],
)
def test_gnina(
    testfile_nolabels, dataroot, device, capsys, model_name, CNNscore, CNNaffinity
):
    args = gnina.options(
        [testfile_nolabels, "-d", dataroot, "--cnn", model_name, "-g", str(device)]
    )

    gnina.main(args)

    captured = capsys.readouterr()
    assert "CNNscore" in captured.out
    assert "CNNaffinity" in captured.out

    score_re = re.findall(r"CNNscore: (.*)", captured.out)
    score = np.array([float(s) for s in score_re])

    affinity_re = re.findall(r"CNNaffinity: (.*)", captured.out)
    affinity = np.array([float(s) for s in affinity_re])

    # CI sometimes fail with 0.43468 instead of 0.43467
    # atol reduced to 1e-5 to avoid this random failure (numerical errors)
    assert np.allclose(1 - score, 1 - CNNscore, atol=1e-5)
    assert np.allclose(affinity, CNNaffinity, atol=1e-5)


@pytest.mark.parametrize(
    "model_name, CNNscore, CNNaffinity, CNNvariance",
    [
        (
            "redock_default2018_ensemble",
            np.array([0.08090, 0.00871, 0.01234]),
            np.array([1.14039, 0.94944, 0.90828]),
            np.array([0.04115, 0.09732, 0.07626]),
        ),
        (
            "general_default2018_ensemble",
            np.array([0.48171, 0.46914, 0.55628]),
            np.array([1.54386, 1.50739, 1.70184]),
            np.array([0.03959, 0.02752, 0.03837]),
        ),
        (
            "crossdock_default2018_ensemble",
            np.array([0.60276, 0.29299, 0.22103]),
            np.array([1.15954, 1.07318, 0.94330]),
            np.array([0.09806, 0.09105, 0.09067]),
        ),
        (
            "dense_ensemble",
            np.array([0.98567, 0.62727, 0.85364]),
            np.array([2.62781, 1.96368, 2.26030]),
            np.array([0.21371, 0.46775, 0.19200]),
        ),
        (
            "default",  # GNINA default model from McNutt et al. (2021)
            np.array([0.66093, 0.43392, 0.44233]),
            np.array([1.82328, 1.49802, 1.54133]),
            np.array([0.56169, 0.21729, 0.38851]),
        ),
    ],
)
def test_gnina_ensemble(
    testfile_nolabels,
    dataroot,
    device,
    capsys,
    model_name,
    CNNscore,
    CNNaffinity,
    CNNvariance,
):
    args = gnina.options(
        [testfile_nolabels, "-d", dataroot, "--cnn", model_name, "-g", str(device)]
    )

    gnina.main(args)

    captured = capsys.readouterr()
    assert "CNNscore" in captured.out
    assert "CNNaffinity" in captured.out
    assert "CNNvariance" in captured.out

    score_re = re.findall(r"CNNscore: (.*)", captured.out)
    score = np.array([float(s) for s in score_re])

    affinity_re = re.findall(r"CNNaffinity: (.*)", captured.out)
    affinity = np.array([float(s) for s in affinity_re])

    variance_re = re.findall(r"CNNvariance: (.*)", captured.out)
    variance = np.array([float(s) for s in variance_re])

    assert np.allclose(1 - score, 1 - CNNscore, atol=1e-5)
    assert np.allclose(affinity, CNNaffinity, atol=1e-5)
    assert np.allclose(1 - variance, 1 - CNNvariance, atol=1e-5)


def test_header(capsys):
    gnina._header()

    captured = capsys.readouterr()

    assert gninatorch.__version__ in captured.out
    assert gninatorch.__git_revision__ in captured.out
