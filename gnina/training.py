import argparse
from typing import List, Optional, Tuple

import molgrid
import numpy as np
import torch
from ignite import metrics
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from torch import nn, optim

from gnina.dataloaders import GriddedExamplesLoader
from gnina.models import models_dict


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
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="Base (initial) learning rate"
    )

    # Misc
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Device name")

    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")

    return parser.parse_args(args)


def _setup_example_provider_and_grid_maker(
    args,
) -> Tuple[molgrid.ExampleProvider, molgrid.GridMaker]:
    """
    Setup :code:`molgrid.ExampleProvider` and :code:`molgrid.GridMaker` based on command
    line arguments.

    Parameters
    ----------
    args: Optional[List[str]]
        List of command line arguments

    Returns
    -------
    Tuple[molgrid.ExampleProvider, molgrid.GridMaker
        Initialized :code:`molgrid.ExampleProvider` and :code:`molgrid.GridMaker`
        dimensions
    """
    example_provider = molgrid.ExampleProvider(
        data_root=args.data_root, balanced=args.balanced, shuffle=args.shuffle
    )
    example_provider.populate(args.trainfile)

    grid_maker = molgrid.GridMaker()

    return example_provider, grid_maker


def training(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        molgrid.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set device
    device = torch.device(args.gpu)

    example_provider, grid_maker = _setup_example_provider_and_grid_maker(args)

    train_loader = GriddedExamplesLoader(
        batch_size=1,
        example_provider=example_provider,
        grid_maker=grid_maker,
        random_translation=0.0,
        random_rotation=False,
    )

    model = models_dict[args.model](train_loader.dims, affinity=False).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr)
    criterion = nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    validation_metrics = {
        "loss": metrics.Loss(criterion),
        "classification": metrics.ClassificationReport(),
    }

    train_evaluator = create_supervised_evaluator(
        model, metrics=validation_metrics, device=device
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        # print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}"
        )
        print(metrics["classification"])

    trainer.run(train_loader, max_epochs=5)


if __name__ == "__main__":
    pass
