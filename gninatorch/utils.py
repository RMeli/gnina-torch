import argparse
import sys
from typing import Optional

import molgrid
import torch


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
    print("", end="", file=stream, flush=True)


def log_print(
    metrics,
    title: Optional[str] = None,
    epoch: Optional[int] = None,
    epoch_time: Optional[float] = None,
    elapsed_time: Optional[float] = None,
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

    if epoch_time is not None:
        print(f"{indent}Epoch Time: {epoch_time:.5f}", file=stream, flush=True)

    if elapsed_time is not None:
        print(f"{indent}Elapsed Time: {elapsed_time:.5f}", file=stream, flush=True)

    # Flush stream
    print("", end="", file=stream, flush=True)


def set_device(device_name: str) -> torch.device:
    """
    Set the device to use.

    Parameters
    ----------
    device_name: str
        Name of the device to use (:code:`"cpu"`, :code:`"cuda"`, :code:`"cuda:0"`, ...)

    Returns
    -------
    torch.device
        PyTorch device

    Notes
    -----
    This function also set the global device for :code:`molgrid` so that the
    :code:`molgrid.ExampoleProvider` works on the correct device.

    https://github.com/gnina/libmolgrid/issues/43
    """
    # TODO: Set global PyTorch device?

    device = torch.device(device_name)
    if "cuda" in device_name:
        try:  # cuda:IDX
            idx = int(device_name[-1])
            molgrid.set_gpu_device(idx)
        except ValueError:  # cuda
            # Set device 0 by default
            molgrid.set_gpu_device(0)

    return device
