import os
import os.path
import pickle
import logging
import re

import torch
import pandas as pd
import numpy as np
import joblib


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


def predict_on_cluster(data, model):
    """
    Given a binary class data (X, y) and a binary classifier model with sklarn API
    * Predict the label
    * Compute the accuracy
    """
    X = data["X"].values
    y_truth = data["y"].values
    y_pred = model.predict(X)
    acc = model.get_metric(X, y_truth)
    return acc


def load_model(path_model):
    """
    Load a sleep_models model and be verbose
    """
    logging.info(f"Loading model {path_model} ...")
    if path_model.endswith(".joblib"):
        model = joblib.load(path_model)
    elif path_model.endswith(".pkl") or path_model.endswith(".pickle"):
        with open(path_model, "rb") as filehandle:
            model = pickle.load(filehandle)
    elif path_model.endswith(".pth"):
        model = torch.load(path_model)
    else:
        raise Exception(
            f"Model file {path_model} not supported. Please use either .joblib, .pkl or .pth"
        )

    logging.info("Done!")
    return model


def read_test_set(folder, cluster):

    return {
        "X": pd.read_csv(os.path.join(folder, f"{cluster}_X-test.csv"), index_col=0,),
        "y": pd.read_csv(os.path.join(folder, f"{cluster}_y-test.csv"), index_col=0,),
    }


def load_and_predict(replicate_folder, clusters, cluster_name):
    """
    Load the model for the cluster_name and predict on all clusters (including itself)
    """
    data = []
    model_files = glob_re(r".*(joblib|pth|pickle)", os.listdir(replicate_folder))
    path_model = [f for f in model_files if cluster_name in f]
    if path_model:
        path_model = os.path.join(replicate_folder, path_model[0])
    else:
        raise Exception(
            f"Model for celltype {cluster_name} not found in {replicate_folder}"
        )

    model = load_model(path_model)

    for versus_cluster_name in clusters:
        logging.info(f"Predicting on {versus_cluster_name} using {cluster_name} model")
        versus_cluster_data = read_test_set(replicate_folder, versus_cluster_name)

        acc = predict_on_cluster(versus_cluster_data, model)
        # _ = versus_cluster_data["X"]
        y = versus_cluster_data["y"]
        f_sleep = y.mean()
        logging.info(f"Accuracy: {acc}, % sleep: {f_sleep}")

        pair = (cluster_name, versus_cluster_name)
        metrics = (acc, f_sleep)

        data.append((pair, metrics))

    return data


def replicate(replicate_folder, background, ncores=10):

    background = pd.read_csv(background, index_col=0, comment="#")["cluster"].tolist()

    if ncores == 1:
        output = []
        for cluster in background:
            output.append(load_and_predict(replicate_folder, background, cluster))
    else:
        output = joblib.Parallel(n_jobs=ncores, verbose=10)(
            joblib.delayed(load_and_predict)(replicate_folder, background, cluster)
            for cluster in background
        )

    accuracy = {cluster: {} for cluster in background}
    fraction_sleep = {cluster: {} for cluster in background}
    for i, full_comparison in enumerate(output):
        for pair in full_comparison:
            (cluster_name, versus_cluster_name), (acc, f_sleep) = pair
            accuracy[cluster_name][versus_cluster_name] = acc
            fraction_sleep[cluster_name][versus_cluster_name] = f_sleep

    # pd.DataFrame creates a table out of a dict of dicts
    # the first level dicts become the columns
    # the second level dicts become the rows
    # therefore, the columns represent the cluster on which the model was trained
    # and the rows represent the clusters on which the model is tested

    ## TRAINED ON #
    ###############
    # R
    # U
    # N
    #
    # O
    # N
    ################

    accuracy_table = pd.DataFrame(accuracy)
    f_sleep_table = pd.DataFrame(fraction_sleep)
    return accuracy_table, f_sleep_table
