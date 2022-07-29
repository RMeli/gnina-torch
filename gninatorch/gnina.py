import argparse
import os
from collections import OrderedDict
from typing import Iterable, List, Optional, Union

import torch
from torch import nn

import gninatorch
from gninatorch import dataloaders, models, setup, utils


def _rename(key: str) -> str:
    """
    Rename GNINA layer to PyTorch layer.

    Parameters
    ----------
    key: str
        GNINA layer name (in loaded state dict)

    Returns
    -------
    str
        PyTorch layer name

    Raises
    ------
    RuntimeError
        if layer name is unknown

    Notes
    -----
    The PyTorch CNN layers are named similarly to the original Caffe layers. However,
    the layer name is prepended with "features.". PyTorch fully connected layers are
    called differently.

    The Default2017 model has slight different naming convention than Default2018 and
    dense models.
    """
    # Fix dense model names
    if "dense_block" in key:
        names = key.split(".")
        return f"features.{names[0]}.blocks.{'.'.join(names[1:])}"
    # Fix non-dense model names (and first data_enc layer)
    elif "conv" in key or "data_enc" in key:
        return f"features.{key}"
    # Fix default2017 model names
    elif "output_fc." in key:
        return key.replace("output_fc", "pose.pose_output")
    elif "output_fc_aff." in key:
        return key.replace("output_fc_aff", "affinity.affinity_output")
    # Fix default2018 and dense models
    elif "pose_output" in key:
        return f"pose.{key}"
    elif "affinity_output" in key:
        return f"affinity.{key}"
    else:  # This should never happen
        raise RuntimeError(f"Unknown layer name: {key}")


def _load_weights(weights_file: str) -> OrderedDict:
    """
    Load weights from file.

    Parameters
    ----------
    weights_file: str
        Path to weights file

    Returns
    -------
    OrderedDict
        Dictionary of weights (renamed according to PyTorch layer names)
    """
    weights = torch.load(weights_file)

    # Rename Caffe layers according to PyTorch names defined in gninatorch.models
    weights_renamed = OrderedDict(
        ((_rename(key), value) for key, value in weights.items())
    )

    return weights_renamed


def _load_gnina_model_file(
    weights_file: str, num_voxels: int
) -> Union[models.Default2017Affinity, models.Default2018Affinity, models.Dense]:
    """
    Load GNINA model from file.

    Parameters
    ----------
    weights_file: str
        Path to weights file
    num_voxels: int
        Number of voxels per grid dimension

    Raises
    ------
    ValueError
        if model name is unknown

    Note
    ----
    All GNINA default models perform both pose prediction and binding affinity
    prediction.
    """
    if "default2017" in weights_file:
        # 32 channels: 18 for the ligand (ligmap.old) and 14 for the protein
        model: Union[
            models.Default2017Affinity, models.Default2018Affinity, models.DenseAffinity
        ] = models.Default2017Affinity(
            input_dims=(35, num_voxels, num_voxels, num_voxels)
        )
    elif "default2018" in weights_file:
        # 28 channels:
        #   14 for the ligand (completelig) and 14 for the protein (completerec)
        model = models.Default2018Affinity(
            input_dims=(28, num_voxels, num_voxels, num_voxels)
        )
    elif "dense" in weights_file:
        # 28 channels:
        #   14 for the ligand (completelig) and 14 for the protein (completerec)
        model = models.DenseAffinity(
            input_dims=(28, num_voxels, num_voxels, num_voxels)
        )
    else:
        raise ValueError(f"Unknown model name: {weights_file}")

    weights = _load_weights(weights_file)
    model.load_state_dict(weights)

    return model


def load_gnina_model(
    gnina_model: str, dimension: float = 23.5, resolution: float = 0.5
):
    """
    Load GNINA model.

    Parameters
    ----------
    gnina_model: str
        GNINA model name
    dimension: float
        Grid dimension (in Angstrom)
    resolution: float
        Grid resolution (in Angstrom)
    """
    path = os.path.dirname(os.path.abspath(__file__))
    gnina_model_file = os.path.join(path, "weights", f"{gnina_model}.pt")

    # Fromhttps://github.com/gnina/libmolgrid/include/libmolgrid/grid_maker.h
    num_voxels = round(dimension / resolution) + 1

    return _load_gnina_model_file(gnina_model_file, num_voxels)


def load_gnina_models(
    model_names: Iterable[str], dimension: float = 23.5, resolution: float = 0.5
):
    """
    Load GNINA models.

    Parameters
    ----------
    model_names: Iterable[str]
        List of GNINA model names
    """
    models_list = []
    for model_name in model_names:
        m = load_gnina_model(model_name, dimension=dimension, resolution=resolution)
        models_list.append(m)

    return models.GNINAModelEnsemble(models_list)


def options(args: Optional[List[str]] = None):
    """
    Define options and parse arguments.

    Parameters
    ----------
    args: Optional[List[str]]
        List of command line arguments
    """
    parser = argparse.ArgumentParser(
        description=" GNINA scoring function",
    )

    parser.add_argument("input", type=str, help="Input file for inference")

    # TODO: Default2017 model needs different ligand types
    parser.add_argument(
        "--cnn",
        type=str,
        help="Pre-trained CNN Model",
        default="default",
        choices=[f"crossdock_default2018{tag}" for tag in ["", "_ensemble"]]
        + [f"crossdock_default2018_{i}" for i in range(1, 5)]
        + [f"general_default2018{tag}" for tag in ["", "_ensemble"]]
        + [f"general_default2018_{i}" for i in range(1, 5)]
        + [f"redock_default2018{tag}" for tag in ["", "_ensemble"]]
        + [f"redock_default2018_{i}" for i in range(1, 5)]
        + [f"dense{tag}" for tag in ["", "_ensemble"]]
        + [f"dense_{i}" for i in range(1, 5)]
        + ["default"],
    )

    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="",
        help="Root folder for relative paths in train files",
    )

    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Device name")

    parser.add_argument("--dimension", type=float, default=23.5, help="Grid dimension")
    parser.add_argument("--resolution", type=float, default=0.5, help="Grid resolution")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    parser.add_argument(
        "--ligmolcache",
        type=str,
        default="",
        help=".molcache2 file for ligands",
    )
    parser.add_argument(
        "--recmolcache",
        type=str,
        default="",
        help=".molcache2 file for receptors",
    )

    parser.add_argument(
        "--no_cache",
        action="store_false",
        help="Disable structure caching",
        dest="cache_structures",
    )

    return parser.parse_args(args)


def setup_gnina_model(
    cnn: str = "default", dimension: float = 23.5, resolution: float = 0.5
) -> Union[nn.Module, bool]:
    """
    Load model or ensemble of models.

    Parameters
    ----------
    cnn: str
        CNN model name
    dimension: float
        Grid dimension
    resolution: float
        Grid resolution

    Returns
    -------
    nn.Module
        Model or ensemble of models

    Notes
    -----
    Mimicks GNINA CLI. The model is returned in evaluation mode. This is essential to
    use the dense model correctly (due to the :code:`nn.BatchNorm` layers).
    """
    ensemble: bool = True

    if cnn == "default":
        # GNINA default model
        # See McNutt et al. J Cheminform (2021) 13:43 for details
        names = [
            "dense",
            "general_default2018_3",
            "dense_3",
            "crossdock_default2018",
            "redock_default2018_2",
        ]

        model = load_gnina_models(names, dimension, resolution)
    elif "ensemble" in cnn:
        ensemble = True

        name = cnn.replace("_ensemble", "")
        names = [name] + [f"{name}_{i}" for i in range(1, 5)]

        # Load model as an ensemble
        model = load_gnina_models(names, dimension, resolution)
    else:
        ensemble = False
        model = load_gnina_model(cnn, dimension, resolution)

    # Put model in evaluation mode
    # This is essential to have the BatchNorm layers in the correct state
    model.eval()

    return model, ensemble


def main(args):
    """
    Run inference with GNINA pre-trained models.

    Parameters
    ----------
    args: Namespace
        Parsed command line arguments

    Notes
    -----
    Models are used in evaluation mode, which is essential for the dense models since
    they use batch normalisation.
    """
    model, ensemble = setup_gnina_model(args.cnn, args.dimension, args.resolution)
    model.eval()  # Ensure models are in evaluation mode!

    device = utils.set_device(args.gpu)
    model.to(device)

    example_provider = setup.setup_example_provider(args.input, args, training=False)
    grid_maker = setup.setup_grid_maker(args)

    # TODO: Allow average over different rotations
    loader = dataloaders.GriddedExamplesLoader(
        example_provider=example_provider,
        grid_maker=grid_maker,
        random_translation=0.0,  # No random translations for inference
        random_rotation=False,  # No random rotations for inference
        device=device,
        grids_only=True,
    )

    for batch in loader:
        if not ensemble:
            log_pose, affinity = model(batch)
        else:
            log_pose, affinity, affinity_var = model(batch)

        pose = torch.exp(log_pose[:, -1])

        for i, (p, a) in enumerate(zip(pose, affinity)):
            print(f"CNNscore: {p:.5f}")
            print(f"CNNaffinity: {a:.5f}")
            if ensemble:
                print(f"CNNvariance: {affinity_var[i]:.5f}")
            print("")


def _header():
    """
    Print GNINA header.

    Notes
    -----
    The header includes an ASCII art logo, and the relevant references.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    logo_file = os.path.join(path, "logo")
    with open(logo_file, "r") as f:
        logo = f.read()

    into_file = os.path.join(path, "intro")
    with open(into_file, "r") as f:
        intro = f.read()

    print(logo, "\n\n", intro)
    print(f"Version: {gninatorch.__version__} ({gninatorch.__git_revision__})\n")


if __name__ == "__main__":
    _header()
    args = options()
    main(args)
