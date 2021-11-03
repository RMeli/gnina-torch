import argparse
import os
from typing import List, Optional

import molgrid
import numpy as np
import torch
from ignite import metrics
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint
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
    parser.add_argument("--testfile", type=str, default=None, help="Test file")
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
        default=None,
        help="Affinity value position in training file",
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, default=os.getcwd(), help="Output directory"
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
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay", default=0.001
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--no_random_rotation",
        action="store_false",
        help="Deactivate random rotation of samples",
        dest="random_rotation",
    )
    parser.add_argument(
        "--random_translation", type=float, default=6.0, help="Random translation"
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=250000,
        help="Number of iterations (epochs)",
    )
    parser.add_argument(
        "--iteration_scheme",
        type=str,
        default="small",
        help="molgrid iteration sheme",
        choices=["small", "large"],
    )

    # Misc
    parser.add_argument(
        "-t", "--test_every", type=int, default=1000, help="Test interval"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Number of epochs per checkpoint",
    )
    parser.add_argument(
        "--num_checkpoints", type=int, default=1, help="Number of checkpoints to keep"
    )
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Device name")

    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")

    return parser.parse_args(args)


_iteration_schemes = {
    "small": molgrid.IterationScheme.SmallEpoch,
    "large": molgrid.IterationScheme.LargeEpoch,
}


def _setup_example_provider(examples_file, args) -> molgrid.ExampleProvider:
    """
    Setup :code:`molgrid.ExampleProvider` based on command line arguments.

    Parameters
    ----------
    args:
        Command line arguments

    Returns
    -------
    molgrid.ExampleProvider
        Initialized :code:`molgrid.ExampleProvider`
    """
    example_provider = molgrid.ExampleProvider(
        data_root=args.data_root,
        balanced=args.balanced,
        shuffle=args.shuffle,
        default_batch_size=args.batch_size,
        iteration_scheme=_iteration_schemes[args.iteration_scheme],
        cache_structs=True,
    )
    example_provider.populate(examples_file)

    return example_provider


def _setup_grid_maker(args) -> molgrid.GridMaker:
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
    grid_maker = molgrid.GridMaker()

    return grid_maker


def _activated_output_transform(output):
    """
    Transform :code:`log_softmax` into probability estimates (i.e. softmax activation).

    https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html#roc-auc
    """
    y_pred, y = output

    # Transform log_softmax output into softmax
    y_pred = torch.exp(y_pred)

    # Return predicted probability only for the positive class
    return y_pred[:, -1], y


def training(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        molgrid.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set device
    device = torch.device(args.gpu)

    # Create example providers
    train_example_provider = _setup_example_provider(args.trainfile, args)
    if args.testfile is not None:
        test_example_provider = _setup_example_provider(args.testfile, args)

    # Create grid maker
    grid_maker = _setup_grid_maker(args)

    train_loader = GriddedExamplesLoader(
        example_provider=train_example_provider,
        grid_maker=grid_maker,
        random_translation=args.random_translation,
        random_rotation=args.random_rotation,
    )

    if args.testfile is not None:
        test_loader = GriddedExamplesLoader(
            example_provider=test_example_provider,
            grid_maker=grid_maker,
            random_translation=args.random_translation,
            random_rotation=args.random_rotation,
        )

        assert test_loader.dims == train_loader.dims

    model = models_dict[args.model](train_loader.dims, affinity=False).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    allmetrics = {
        # Balanced accuracy is the average recall over all classes
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
        "balanced_accuracy": metrics.Recall(average=True),
        # Accuracy can be used directly without binarising the data since we are not
        # performing binary classification (Linear(out_features=1)) but we are
        # performing multiclass classification with 2 classes (Linear(out_features=2))
        "accuracy": metrics.Accuracy(),
        "classification": metrics.ClassificationReport(),
        "roc_auc": ROC_AUC(output_transform=_activated_output_transform),
    }

    evaluator = create_supervised_evaluator(model, metrics=allmetrics, device=device)

    # FIXME: This requires a second pass on the training set
    # FIXME: Measures should be accumulated: https://pytorch.org/ignite/quickstart.html#f1
    @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f">>> Train Results - Epoch[{trainer.state.epoch}] <<<")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2f}")
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")

    if args.testfile is not None:

        @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
        def log_test_results(trainer):
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            print(f">>> Test Results - Epoch[{trainer.state.epoch}] <<<")
            print(f"Accuracy: {metrics['accuracy']:.2f}")
            print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2f}")
            print(f"ROC AUC: {metrics['roc_auc']:.2f}", flush=True)
            # print(metrics["classification"])

    # TODO: Save input parameters as well
    to_save = {"model": model, "optimizer": optimizer}
    checkpoint = Checkpoint(
        to_save,
        args.out_dir,
        n_saved=args.num_checkpoints,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=args.checkpoint_every), checkpoint
    )

    if args.progress_bar:
        pbar = ProgressBar()
        pbar.attach(trainer)

    trainer.run(train_loader, max_epochs=args.iterations)


if __name__ == "__main__":
    args = options()
    training(args)
