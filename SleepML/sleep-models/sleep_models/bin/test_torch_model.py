import logging
import pickle
import os.path
import numpy as np
import pandas as pd
import torch.cuda
from torch import nn

from sleep_models.bin.train_model import setup_logging
from sleep_models.bin.train_model import base_parser
from sleep_models.models import MODELS

from sleep_models.predict import read_test_set

# from sleep_models.models.utils.plotting import load_and_plot_confusion_table
from sleep_models.preprocessing import make_confusion_long_to_square
from sleep_models.plotting import plot_confusion_table


def select_model(model_name):
    return [model for model in MODELS if model.__name__ == model_name][0]


def test_model(
    input,
    cluster,
    cluster_data,
    random_state=1000,
    verbose=logging.WARNING,
    logfile=None,
    fraction=1.0,
):

    """
    Arguments:

        input (str): Path to a folder where the results will be saved.
            On this folder, a new folder will be created with name random-state-{random_state}
        cluster (str): CellType on which the model has been trained
        cluster_data (str): CelTypes to test the model against (not necessarily the same as cluster)
        random_state (int): random random_state for reproducibility

    Returns: None
    """
    if logfile is None:
        logfile = os.path.join("logs", f"test_model_{cluster}.log")

    logger, _ = setup_logging(verbose, logfile)
    logger.info(f"Training on cell type {cluster} starting!")
    input = os.path.join(input, f"random-state_{random_state}_fraction_{str(fraction)}")

    model_filename = f"{cluster}.pth"
    model_path = os.path.join(input, model_filename)
    model = torch.load(model_path)

    if cluster != cluster_data:
        data_folder = input.replace(cluster, cluster_data)
    else:
        data_folder = input

    test_set = read_test_set(data_folder, cluster_data)
    X_test = test_set["X"]
    y_test = test_set["y"]

    predictions = model.predict(X_test.values)
    truth = y_test.values.argmax(1)

    confusion_table = model.get_confusion_table(truth, predictions)
    plot_confusion_table(
        confusion_table,
        os.path.join(input, f"{cluster_data}-test_confusion_table.png"),
    )

    accuracy = model.score(X_test.values, y_test.values)

    with open(os.path.join(input, f"{cluster_data}_metrics.yaml"), "w") as filehandle:
        filehandle.write(f"test-accuracy: " f"{str(accuracy)}\n")


def get_accuracy_by_label(labels, agreement):

    accuracy = {}

    for i, label in enumerate(labels):
        if label not in accuracy:
            accuracy[label] = [0, 0]

        accuracy[label][agreement[i]] += 1

    print("Accuracy by label: ")
    print(accuracy)
    return accuracy


def get_parser(ap=None, *args, **kwargs):
    ap = base_parser(ap, *args, **kwargs)
    ap.add_argument("--cluster-data", dest="cluster_data", type=str, required=True)
    ap.add_argument("--fraction", dest="fraction", type=float, default=1.0)
    return ap


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap=ap, models=MODELS)
        args = ap.parse_args()

    result = test_model(
        input=args.output,
        cluster=args.cluster,
        cluster_data=args.cluster_data,
        random_state=args.random_state,
        fraction=args.fraction,
    )


if __name__ == "__main__":
    main()
