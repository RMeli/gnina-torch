import argparse
import sys
from typing import Optional


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
