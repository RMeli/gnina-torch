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
    loss: float = 0.0
    for name, value in metrics.items():
        print(f"{indent}{name}: {value:.5f}", file=stream)
        if "loss" in name.lower():
            loss += value

    if loss > 0:
        print(f"    Loss: {loss:.5f}", file=stream)
    print(flush=True)
