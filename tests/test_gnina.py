import molgrid
import numpy as np
import pytest
import torch

from gnina import gnina, models
from gnina.dataloaders import GriddedExamplesLoader


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


@pytest.mark.xfail(reason="Not implemented")
def test_load_gnina_model_dense():
    raise NotImplementedError


def test_load_gnina_model_wrong():
    with pytest.raises(ValueError, match="Unknown model name"):
        gnina.load_gnina_model("wrong_model_name")


def test_gnina_model_prediction_default2018(dataroot, testfile, device):
    model = gnina.load_gnina_model("redock_default2018")
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

    # Select scores of the negative class
    # This is mainly because for the fictitious test systems the score is really low
    # Small scores (close to zero) do not play nicely with np.allclose
    negative_score = torch.exp(log_pose)[:, 0].cpu().numpy()

    # Scores obtained with GNINA
    # https://github.com/gnina/gnina
    #   gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l1.sdf --score_only --cnn redock_default2018
    #   gnina -r tests/data/mols/r2.pdb -l tests/data/mols/l2.sdf --score_only --cnn redock_default2018
    #   gnina -r tests/data/mols/r1.pdb -l tests/data/mols/l2.sdf --score_only --cnn redock_default2018

    assert np.allclose(negative_score, 1 - np.array([0.02956, 0.00114, 0.00095]))
    assert np.allclose(affinity.cpu().numpy(), np.array([1.31840, 1.20986, 1.14063]))
