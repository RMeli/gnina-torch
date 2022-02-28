import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

import molgrid
import numpy as np
import pandas as pd
import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint

from gnina import metrics, models, setup, training, utils
from gnina.dataloaders import GriddedExamplesLoader


def options(args: Optional[List[str]] = None):
    """
    Define options and parse arguments.

    Parameters
    ----------
    args: Optional[List[str]]
        List of command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Inference with GNINA scoring function",
    )

    parser.add_argument("input", type=str, help="Input file for inference")
    parser.add_argument("model", type=str, help="Model")
    parser.add_argument("checkpoint", type=str, help="Checkpoint file")

    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="",
        help="Root folder for relative paths in train files",
    )

    parser.add_argument(
        "--rotations",
        type=int,
        default=1,
        help="Number of rotations to average on",
    )

    parser.add_argument(
        "-o", "--out_dir", type=str, default=os.getcwd(), help="Output directory"
    )

    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Device name")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")

    # TODO: Retrieve the following parameters from the chekpoint file!
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
    parser.add_argument("--dimension", type=float, default=23.5, help="Grid dimension")
    parser.add_argument("--resolution", type=float, default=0.5, help="Grid resolution")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    # Misc
    parser.add_argument("--silent", action="store_true", help="No console output")
    parser.add_argument(
        "--no_roc_auc",
        action="store_false",
        help="Disable ROC AUC (useful for crystal poses)",
        dest="roc_auc",
    )
    parser.add_argument(
        "--no_csv",
        action="store_false",
        help="Disable CSV output",
        dest="csv",
    )

    return parser.parse_args(args)


def inference(args):
    """
    Main function for inference with GNINA scoring function.

    Parameters
    ----------
    args:
    """

    # Create necessary directories if not already present
    os.makedirs(args.out_dir, exist_ok=True)

    # Define output streams for logging
    logfile = open(os.path.join(args.out_dir, "inference.log"), "w")
    if not args.silent:
        outstreams = [sys.stdout, logfile]
    else:
        outstreams = [logfile]

    # Print command line arguments
    for outstream in outstreams:
        utils.print_args(args, "--- GNINA INFERENCE ---", stream=outstream)

    # Set random seed for reproducibility
    if args.seed is not None:
        molgrid.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set device
    device = utils.set_device(args.gpu)

    # Create example providers
    test_example_provider = setup.setup_example_provider(
        args.input, args, training=False
    )

    # Create grid maker
    grid_maker = setup.setup_grid_maker(args)

    test_loader = GriddedExamplesLoader(
        example_provider=test_example_provider,
        grid_maker=grid_maker,
        label_pos=args.label_pos,
        affinity_pos=args.affinity_pos,
        random_translation=0.0,  # No random translations for inference
        random_rotation=False,  # No random rotations for inference
        device=device,
    )

    affinity: bool = True if args.affinity_pos is not None else False

    # Create model
    model = models.models_dict[(args.model, affinity)](test_loader.dims).to(device)

    # Compile model with TorchScript
    model = torch.jit.script(model)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    # TODO: Allow prediction for systems without known pose or affinity
    # Setup metrics but do not compute losses
    allmetrics = metrics.setup_metrics(
        affinity,
        pose_loss=None,
        affinity_loss=None,
        roc_auc=args.roc_auc,
        device=device,
    )
    evaluator = training._setup_evaluator(model, allmetrics, affinity=affinity)

    results = defaultdict(list)
    metrics_inference = defaultdict(list)

    # Print predictions for every batch
    # evaluator.state.output only stores the last batch
    @evaluator.on(Events.ITERATION_COMPLETED)
    def print_output(evaluator):
        output = evaluator.state.output

        # Extract probability of good pose only
        pose_pred = torch.exp(output["pose_log"])[:, -1]
        assert pose_pred.shape == output["labels"].shape

        results["pose_prob"] = np.concatenate(
            (results["pose_prob"], pose_pred.cpu().numpy())
        )
        results["pose_label"] = np.concatenate(
            (results["pose_label"], output["labels"].cpu().numpy())
        )

        try:
            # This fails with KeyError if affinity is not present
            assert output["affinities_pred"].shape == output["affinities"].shape
            assert output["affinities_pred"].shape == output["labels"].shape

            results["affinity_pred"] = np.concatenate(
                (results["affinity_pred"], output["affinities_pred"].cpu().numpy())
            )
            # Return absolute binding affinity
            # Experimental values are negative for a bad pose
            results["affinity_exp"] = np.concatenate(
                (results["affinity_exp"], np.abs(output["affinities"].cpu().numpy()))
            )
        except KeyError:
            # No binding affinity prediction available
            pass

    evaluator.run(test_loader)

    for outstream in outstreams:
        utils.log_print(
            evaluator.state.metrics,
            stream=outstream,
        )

    if args.csv:
        pd.DataFrame(results).to_csv(
            os.path.join(args.out_dir, "inference.csv"), float_format="%.5f"
        )

        for key, value in evaluator.state.metrics.items():
            metrics_inference[key].append(value)

        pd.DataFrame(metrics_inference).to_csv(
            os.path.join(args.out_dir, "metrics_inference.csv"),
            float_format="%.5f",
            index=False,
        )

    # Close log file
    logfile.close()


if __name__ == "__main__":
    args = options()
    inference(args)
