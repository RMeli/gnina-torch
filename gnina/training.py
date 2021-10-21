import argparse
from typing import List, Optional, Tuple

import molgrid
import numpy as np
import torch

# from gnina.models import models_dict


def options(args: Optional[List[str]] = None):
    """
    Define options and parse arguments.

    Parameters
    ----------
    args: Optional[List[str]]
        List of command line arguments
    """
    parser = argparse.ArgumentParser(
        description="GNINA scoring function",
    )

    # Data
    # TODO: Allow multiple train files?
    parser.add_argument("trainfile", type=str, help="Training file")
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="",
        help="Root folder for relative paths in train files",
    )
    parser.add_argument(
        "--balanced", action="store_true", help="Balanced sampling of receptors"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_false",
        help="Deactivate random shuffling of samples",
        dest="shuffle",  # Variable name (shuffle is False when --no_shuffle is used)
    )
    parser.add_argument(
        "--label_pos", type=int, default=0, help="Pose label position in training file"
    )
    parser.add_argument(
        "--affinity_pos",
        type=int,
        default=1,
        help="Affinity value position in training file",
    )

    # Scoring function
    parser.add_argument(
        "-m", "--model", type=str, default="default2017", help="Model name"
    )
    # TODO: ligand type file and receptor type file (default: 28 types)

    # Learning

    # Misc
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Device name")

    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")

    return parser.parse_args(args)


def _setup_example_provider_and_grid_maker(
    args,
) -> Tuple[molgrid.ExampleProvider, molgrid.GridMaker, Tuple[int]]:
    """
    Setup :code:`molgrid.ExampleProvider` and :code:`molgrid.GridMaker` based on command
    line arguments.

    Parameters
    ----------
    args: Optional[List[str]]
        List of command line arguments

    Returns
    -------
    Tuple[molgrid.ExampleProvider, molgrid.GridMaker, Tuple[int]]
        Initialized :code:`molgrid.ExampleProvider`, :code:`molgrid.GridMaker` and grid
        dimensions
    """
    example_provider = molgrid.ExampleProvider(
        data_root=args.data_root, balanced=args.balanced, shuffle=args.shuffle
    )
    example_provider.populate(args.trainfile)

    grid_maker = molgrid.GridMaker()
    dims = grid_maker.grid_dimensions(example_provider.num_types())

    return example_provider, grid_maker, dims


def training(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        molgrid.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set device
    # device = torch.device(args.gpu)

    example_provider, grid_maker, dims = _setup_example_provider_and_grid_maker(args)

    # model = models_dict[args.model](dims)


if __name__ == "__main__":
    pass
