import argparse
import os
import os.path
import logging

import pandas as pd
import numpy as np
import joblib

from sleep_models.predict import replicate

restore_cache = False
logger = logging.getLogger(__name__)


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--background",
        type=str,
        help="This file tells the program which clusters should the current cluster be compared to",
        required=True,
    )
    ap.add_argument(
        "--training-output",
        "--input",
        type=str,
        dest="training_output",
        help="Folder where the replicate data splits+train have been saved",
        required=True,
    )
    ap.add_argument(
        "--ncores",
        "-j",
        type=int,
        dest="ncores",
        help="Number of parallel CPUs to use",
        default=1,
    )
    return ap


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    predict(
        training_output=args.training_output,
        prediction_output=args.training_output,
        ncores=args.ncores,
    )


def predict(training_output, prediction_output, ncores=1):

    background_file = os.path.join(training_output, "background.txt")

    with open(background_file, "r") as fh:
        background = os.path.join(fh.readline().strip("\n"))

    clusters = pd.read_csv(background, index_col=False, comment="#")["cluster"].values

    folders = sorted(os.listdir(training_output))
    folders = [f for f in folders if f.startswith("random-state")]

    if not restore_cache:
        results = {}
        for replicate_folder in folders:

            # run an independent replicate
            # NOTE: This can take a while to run, especially when crossing a lot of models
            accuracy_table, f_sleep_table = replicate(
                os.path.join(training_output, replicate_folder),
                clusters,
                ncores=ncores,
            )
            # save to disk
            accuracy_table.to_csv(
                os.path.join(prediction_output, replicate_folder, "accuracy.csv")
            )
            f_sleep_table.to_csv(
                os.path.join(prediction_output, replicate_folder, "f_sleep.csv")
            )
            # store in RAM
            results[replicate_folder] = (accuracy_table, f_sleep_table)

    else:
        # not implemented
        raise NotImplementedError("Restoring from cache is not implemented")

    return 0


if __name__ == "__main__":
    main()
