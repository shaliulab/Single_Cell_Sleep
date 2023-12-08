import json
import os
import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_validate

from sleep_models.utils.adata import restore_dataset
from sleep_models.models import MODELS


logger = logging.getLogger("sleep_models.sweep")


def mean_squared_error(model, X, y):

    pred = model.predict(X)
    truth = y.values

    loss = np.mean((pred - truth) ** 2)
    return loss


def _train(run, config):
    """
    wandb-independent training logic
    """

    self = model

    model_kwargs = self._model_kwargs.copy()
    model_kwargs.update(self.process_wandb_config(config))

    print("Making model...")

    ModelClass = MODELS[config.model_name]

    model = ModelClass(
        cluster=self._cluster,
        results_dir=self._results_dir,
        random_state=self.random_state,
        **model_kwargs,
    )

    print("Preparing dataset...")
    X_train, y_train, X_test, y_test = restore_dataset(cluster=self._cluster)

    print("Training...")
    model.fit(X_train, y_train, X_test, y_test)

    loss = model.get_loss(X_train, y_train)
    test_loss = model.get_loss(X_test, y_test)

    print("Computing performance metrics...")
    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    corr, p = pearsonr(model.y_train.values.flatten(), y_pred.flatten())
    corr_test, p_test = pearsonr(model.y_test.values.flatten(), y_pred_test.flatten())

    msqe = -cross_validate(
        model, X_test, y_test, cv=3, scoring="neg_mean_squared_error",
    )["test_score"].mean()

    metric = model.get_metric(X_train, y_train)
    test_metric = model.get_metric(X_test, y_test)

    print("Model performance:")
    tolog = {
        "loss": loss,
        "test_loss": test_loss,
        model._metric.lower(): metric,
        f"test_{model._metric.lower()}": test_metric,
        "rho": corr,
        "rho_test": corr_test,
        "p": p,
        "p_test": p_test,
        "mean_squared_error": msqe,
    }

    return tolog


def train(ModelClass):
    """
    wandb logic required for training
    """

    with wandb.init("Sweep") as run:
        tolog = _train(ModelClass=ModelClass, run=run, config=wandb.config)
        print(tolog)
        wandb.log(tolog)


def sweep(model, config_file, sweeps=100):

    with open(config_file, "r") as fh:
        config = json.load(fh)

    sweep_id = wandb.sweep(config)
    wandb.agent(sweep_id, function=train, count=sweeps)


@staticmethod
def process_wandb_config(config):

    if isinstance(config, dict):
        pass
    else:
        config = config.__dict__["_items"]

    # config = config["parameters"].copy()
    # for k, v in config.items():
    #     if "min" in v.keys():
    #         config[k] = v["min"]
    #     elif "values" in v.keys():
    #         config[k] = v["values"][0]

    config["hidden_layer_sizes"] = (config.pop("hidden_layer_size"),) * config.pop(
        "n_layers"
    )

    config.pop("_wandb")

    return config
