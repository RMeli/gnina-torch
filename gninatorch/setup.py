"""
Utility functions to setup training and inference.
"""

import argparse

import molgrid

# TODO: Ensure that for inference all systems are seen exactly once?
_iteration_schemes = {
    "small": molgrid.IterationScheme.SmallEpoch,
    "large": molgrid.IterationScheme.LargeEpoch,
}


def setup_example_provider(
    examples_file, args: argparse.Namespace, training: bool = True
) -> molgrid.ExampleProvider:
    """
    Setup :code:`molgrid.ExampleProvider` based on command line arguments.

    Parameters
    ----------
    examples_file: str
        File with examples (.types file)
    args: argparse.Namespace
        Command line arguments
    train: bool
        Flag to distinguis between training and inference

    Returns
    -------
    molgrid.ExampleProvider
        Initialized :code:`molgrid.ExampleProvider`

    Notes
    -----
    For inference (:code:`training=False`), the data set is not balanced, stratified,
    nor shuffled.
    """
    if training:
        # Use command line option for training
        example_provider = molgrid.ExampleProvider(
            data_root=args.data_root,
            balanced=args.balanced,
            shuffle=args.shuffle,
            default_batch_size=args.batch_size,
            iteration_scheme=_iteration_schemes[args.iteration_scheme],
            ligmolcache=args.ligmolcache,
            recmolcache=args.recmolcache,
            stratify_receptor=args.stratify_receptor,
            stratify_pos=args.stratify_pos,
            stratify_max=args.stratify_max,
            stratify_min=args.stratify_min,
            stratify_step=args.stratify_step,
            cache_structs=args.cache_structures,
        )
    else:
        # Use command line option for training
        example_provider = molgrid.ExampleProvider(
            data_root=args.data_root,
            balanced=False,
            shuffle=False,
            default_batch_size=args.batch_size,
            iteration_scheme=_iteration_schemes["small"],
            ligmolcache=args.ligmolcache,
            recmolcache=args.recmolcache,
            stratify_receptor=False,
            cache_structs=args.cache_structures,
        )

    example_provider.populate(examples_file)

    return example_provider


def setup_grid_maker(args) -> molgrid.GridMaker:
    """
    Setup :code:`molgrid.ExampleProvider` and :code:`molgrid.GridMaker` based on command
    line arguments.

    Parameters
    ----------
    args:
        Command line arguments

    Returns
    -------
    molgrid.GridMaker
        Initialized :code:`molgrid.GridMaker`
    """
    grid_maker = molgrid.GridMaker(resolution=args.resolution, dimension=args.dimension)

    return grid_maker
