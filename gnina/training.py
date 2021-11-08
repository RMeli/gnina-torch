import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import molgrid
import numpy as np
import torch
from ignite import metrics
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import Checkpoint
from torch import nn, optim

from gnina.dataloaders import GriddedExamplesLoader
from gnina.losses import PseudoHuberLoss
from gnina.models import models_dict

_iteration_schemes = {
    "small": molgrid.IterationScheme.SmallEpoch,
    "large": molgrid.IterationScheme.LargeEpoch,
}


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
        "-m",
        "--model",
        type=str,
        default="default2017",
        help="Model name",
        choices=models_dict.keys(),
    )
    parser.add_argument("--dimension", type=float, default=23.5, help="Grid dimension")
    parser.add_argument("--resolution", type=float, default=0.5, help="Grid resolution")
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
        choices=_iteration_schemes.keys(),
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


def _setup_example_provider(examples_file, args) -> molgrid.ExampleProvider:
    """
    Setup :code:`molgrid.ExampleProvider` based on command line arguments.

    Parameters
    ----------
    examples_file: str
        File with examples (.types file)
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
    grid_maker = molgrid.GridMaker(resolution=args.resolution, dimension=args.dimension)

    return grid_maker


def _activated_output_transform(output):
    """
    Transform :code:`log_softmax` into probability estimates (i.e. softmax activation).

    Parameters
    ----------
    output:
        Output of :code:`nn.Module`

    Returns
    -------
    Tuple[torch.Tensor]
        Probability estimates for the positive class and true label.

    Notes
    -----
    https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html#roc-auc
    """
    y_pred, y = output

    # Transform log_softmax output into softmax
    y_pred = torch.exp(y_pred)

    # Return predicted probability only for the positive class
    return y_pred[:, -1], y


def _train_step_pose_and_affinity(
    trainer: Engine, batch, model, optimizer, pose_loss, affinity_loss
):
    """
    Update for pose and affinity prediction.
    """
    model.train()
    optimizer.zero_grad()

    # Data is already on the correct device thanks to the ExampleProvider
    grids, labels, affinities = batch

    pose_log, affinities_pred = model(grids)

    # Compute combined loss for pose prediction and affinity prediction
    loss = pose_loss(pose_log, labels) + affinity_loss(affinities_pred, affinities)

    loss.backward()
    # TODO: Gradient clipping
    optimizer.step()

    return loss.item()


def _setup_trainer(model, optimizer, device, affinity: bool = False) -> Engine:
    """
    Setup training engine for binding pose prediction or binding pose and affinity
    prediction.

    Patameters
    ----------
    model:
        Model to train
    optimizer:
        Optimizer
    device: torch.device
        Device
    affinity: bool
        Flag for affinity prediction (in addition to pose prediction)

    Notes
    -----
    If :code:`affinity==True`, the model return both pose and affinity predictions,
    which requites a custom training step to evaluate the combine loss function. The
    custom training step is defined in :fun:`_train_step_pose_and_affinity`.
    """
    if affinity:
        # Pose prediction and binding affinity prediction
        # Create engine based on custom train step
        trainer = Engine(
            lambda trainer, batch: _train_step_pose_and_affinity(
                trainer,
                batch,
                model,
                optimizer,
                pose_loss=nn.NLLLoss(),
                affinity_loss=PseudoHuberLoss(delta=4.0),
            )
        )
    else:
        # Pose prediction only
        trainer = create_supervised_trainer(model, optimizer, nn.NLLLoss(), device)

    return trainer


def _evaluation_step_pose_and_affinity(evaluator: Engine, batch, model):
    """
    Evaluate model for binding pose and affinity prediction.

    Parameters
    ----------
    evaluator:
        PyTorch Ignite :code:`Engine`
    batch:
        Batch data
    model:
        Model

    Returns
    -------
    Tuple[torch.Tensor]
        Class probabilities for pose prediction, affinity prediction, true pose labels
        and experimental binding affinities

    Notes
    -----
    The model returns the softmax of the last linear layer for binding pose prediction
    (class probabilities) and the raw output of the last linear layer for binding
    affinity prediction, together with the pose labels and experimental binding
    affinities.
    """
    model.eval()
    with torch.no_grad():
        grids, labels, affinities = batch
        pose_log, affinities_pred = model(grids)

    output = {
        "pose_log": pose_log,
        "affinities_pred": affinities_pred,
        "labels": labels,
        "affinities": affinities,
    }

    return output


def _setup_evaluator(model, metrics, device, affinity: bool = False) -> Engine:
    """
    Setup PyTorch Ignite :code:`Engine` for evaluation.

    Parameters
    ----------
    model:
        PyTorch model
    metrics:
        Evaluation metrics
    device: torch.device
        Device
    affinity: bool
        Flag for affinity prediction (in addition to pose prediction)

    Returns
    -------
    ignite.Engine
        PyTorch Ignite engine for evaluation

    Notes
    -----
    For pose prediction the model is rather standard (single outpout) and therefore
    the :code:`create_supervised_evaluator()` factory function is used. For both pose
    and binding affinity prediction, the custom
    :code:`_evaluation_step_pose_and_affinity` is used instead.
    """
    if affinity:
        evaluator = Engine(
            lambda evaluator, batch: _evaluation_step_pose_and_affinity(
                evaluator, batch, model
            )
        )

        # Add metrics to the evaluator engine
        # Metrics need an output_tranform method in order to select the correct ouput
        # from _evaluation_step_pose_and_affinity
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    else:
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    return evaluator


# def _output_transform_pose_and_affinity(x, y, y_pred):
#    """
#    Output transformation for affinity prediction.
#    """
#    labels, affinities = y
#    pose_log, affinities_pred = y_pred

#    return pose_log, affinities_pred, labels, affinities


def _output_transform_identity(args: Tuple[Any]) -> Tuple[Any]:
    """
    Output transformation that does nothing.

    Parameters
    ----------
    args: Tuple[Any]
        Tuple of arguments

    Returns
    -------
    Tuple[Any]
        Tuple of arguments unchanged

    Notes
    -----
    Identity transformation when an :code:`output_transform` function is not needed
    (default behaviour works well for single output model for pose prediction).
    """
    return args


def _output_transform_select_pose(
    output: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Notes
    -----
    THis function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select pose results from
    what the evaluator returns (that is,
    :code:`(pose_log, affinities_pred, labels, affinities)` when :code:`affinity=True`).
    See return of :fun:`_output_transform_pose_and_affinity`.

    The output is activated, i.e. the :code:`log_softmax` output is transformed into
    :code:`softmax`.
    """
    # Return pose class probabilities and true labels
    # log_softmax is transformed into softmax to get the class probabilities
    return torch.exp(output["pose_log"]), output["labels"]


def _output_transform_select_affinity(
    output: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Predicted binding affnity and experimental binding affinity

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select affinity predictions from
    what the evaluator returns (that is,
    :code:`(pose_log, affinities_pred, labels, affinities)` when :code:`affinity=True`).
    See return of :fun:`_output_transform_pose_and_affinity`.
    """
    # Return pose class probabilities and true labels
    return output["affinities_pred"], output["affinities"]


def _output_transform_ROC(output, affinity: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Output transform for the ROC curve.

    Parameters
    ----------
    output:
        Engine output
    affinity: bool
        Flag for binding affinity prediction

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Positive class probabilities and associated labels.

    Notes
    -----
    https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html#roc-auc
    """
    if affinity:
        # Select pose prediction if binding affinity is predicted as well
        pose, labels = _output_transform_select_pose(output)
    else:
        pose, labels = output

    # Return probability estimates of the positive class
    return pose[:, -1], labels


def _setup_metrics(affinity: bool, device):
    # Pose prediction metrics
    m = {
        # Balanced accuracy is the average recall over all classes
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
        "balanced_accuracy": metrics.Recall(
            average=True,
            output_transform=_output_transform_select_pose
            if affinity
            else _output_transform_identity,
        ),
        # Accuracy can be used directly without binarising the data since we are not
        # performing binary classification (Linear(out_features=1)) but we are
        # performing multiclass classification with 2 classes (Linear(out_features=2))
        "accuracy": metrics.Accuracy(
            output_transform=_output_transform_select_pose
            if affinity
            else _output_transform_identity
        ),
        # "classification": metrics.ClassificationReport(),
        "roc_auc": ROC_AUC(
            output_transform=lambda output: _output_transform_ROC(
                output, affinity=affinity
            ),
            device=device,
        ),
    }

    # Affinity prediction metrics
    if affinity:
        m.update(
            {
                "MAE": metrics.MeanAbsoluteError(
                    output_transform=_output_transform_select_affinity
                ),
                "MSE": metrics.MeanSquaredError(
                    output_transform=_output_transform_select_affinity
                ),
            }
        )

    # Return dictionary with all metrics
    return m


def _log_print(title, epoch, metrics, affinity: bool):
    print(f">>> {title} - Epoch[{epoch}] <<<")

    # Pose classification metriccs
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2f}")
    print(f"ROC AUC: {metrics['roc_auc']:.2f}", flush=True)

    # Binding affinity prediction metrics
    if affinity:
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"MSE: {metrics['MSE']:.2f}")


def training(args):
    """
    Main function for training GNINA scoring function.

    Parameters
    ----------
    args:
        Command line arguments
    """
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
        label_pos=args.label_pos,
        affinity_pos=args.affinity_pos,
        random_translation=args.random_translation,
        random_rotation=args.random_rotation,
        device=device,
    )

    if args.testfile is not None:
        test_loader = GriddedExamplesLoader(
            example_provider=test_example_provider,
            grid_maker=grid_maker,
            label_pos=args.label_pos,
            affinity_pos=args.affinity_pos,
            random_translation=args.random_translation,
            random_rotation=args.random_rotation,
            device=device,
        )

        assert test_loader.dims == train_loader.dims

    affinity: bool = True if args.affinity_pos is not None else False

    # Create model
    model = models_dict[args.model](train_loader.dims, affinity=affinity).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    trainer = _setup_trainer(model, optimizer, device, affinity=affinity)

    allmetrics = _setup_metrics(affinity, device)

    evaluator = _setup_evaluator(model, allmetrics, device, affinity=affinity)

    # FIXME: This requires a second pass on the training set
    # FIXME: Measures should be accumulated: https://pytorch.org/ignite/quickstart.html#f1
    @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
    def log_training_results(trainer):
        evaluator.run(train_loader)
        _log_print(
            "Train Results",
            trainer.state.epoch,
            evaluator.state.metrics,
            affinity=affinity,
        )

    if args.testfile is not None:

        @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
        def log_test_results(trainer):
            evaluator.run(test_loader)
            _log_print(
                "Test Results",
                trainer.state.epoch,
                evaluator.state.metrics,
                affinity=affinity,
            )

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
