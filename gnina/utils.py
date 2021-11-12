import argparse
import sys
from typing import Optional

from torch import nn


def print_args(
    args: argparse.Namespace, header: Optional[str] = None, stream=sys.stdout
):
    """
    Print command line arguments to stream.py

    Parameters
    ----------
    args: argparse.Namespace
        Command line arguments
    header: str

    stream:
        Output stream
    """
    if header is not None:
        print(header, file=stream)
    for name, value in vars(args).items():
        if type(value) is float:
            print(f"{name}: {value:.5E}", file=stream)
        else:
            print(f"{name} = {value!r}", file=stream)

    # Flush stream
    print("", file=stream, flush=True)


def log_print(
    metrics,
    output=None,
    title: Optional[str] = None,
    epoch: Optional[int] = None,
    pose_loss: Optional[nn.Module] = None,
    affinity_loss: Optional[nn.Module] = None,
    stream=sys.stdout,
):
    """
    Print metrics to the console.

    Parameters
    ----------
    metrics:
        Dictionary of metrics
    output:
        Engine output
    title: str
        Title to print
    epoch: int
        Epoch number
    pose_loss: nn.Module
        Pose loss
    affinity_loss: nn.Module
        Affinity loss
    stream:
        Outoput stream
    """
    if title is not None and epoch is not None:
        print(f">>> {title} - Epoch[{epoch}] <<<", file=stream)
        indent = "    "
    else:
        indent = ""

    # TODO: Order metrics?
    for name, value in metrics.items():
        print(f"{indent}{name}: {value:.5f}", file=stream)

    if output is not None:
        loss = 0
        if pose_loss is not None:
            pl = pose_loss(output["pose_log"], output["labels"])
            print(f"{indent}Loss (pose): {pl:.5f}", file=stream)
            loss += pl.item()

        if affinity_loss is not None:
            al = affinity_loss(output["affinities_pred"], output["affinities"])
            print(f"{indent}Loss (affinity): {al:.5f}", file=stream)
            loss += al.item()

        if loss > 0:
            print(f"    Loss: {loss:.5f}", file=stream)

    print(flush=True)
