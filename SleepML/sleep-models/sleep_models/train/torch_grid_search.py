import json
import traceback
import logging
import os.path
from datetime import datetime
import socket

import numpy as np
import matplotlib.pyplot as plt
import wandb

from sleep_models import __version__ as SLEEP_MODELS_VERSION
from sleep_models.models.core import train
from sleep_models.utils.data import restore_dataset
from sleep_models.models.variables import AllConfig
import sleep_models.models.utils.torch as torch_utils
from sleep_models.models import MODELS
import sleep_models.preprocessing as pp
import sleep_models.utils.data as data_utils
from sleep_models.models.variables import CONFIGS

logger = logging.getLogger("sleep_models.sweep")


def get_sweep_name(cluster, model_name):
    machine_name = socket.gethostname()
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_name = f"{cluster}/{machine_name}/{model_name}/{datetime_now}"
    return sweep_name


def train_wandb():
    """
    wandb logic required for training
    """
    device = torch_utils.get_device()
    with wandb.init(save_code=True, notes=SLEEP_MODELS_VERSION) as run:

        ModelClass = MODELS[wandb.config.model_name]

        if (
            os.path.isfile(wandb.config.input)
            and wandb.config.input.split(".")[-1] == "h5ad"
        ):

            data = pp.load_data(
                h5ad_input=wandb.config.input,
                output=output,
                cluster=wandb.config.cluster,
                random_state=wandb.config.random_state,
                mean_scale=wandb.config.mean_scale,
                highly_variable_genes=wandb.config.highly_variable_genes,
                label_mapping=wandb.config.label_mapping,
                model_properties=ModelClass.model_properties(),
                fraction=wandb.config.fraction,
            )
            X_train, y_train, _, _ = data["datasets"]

        elif os.path.isdir(wandb.config.input):
            X_train, y_train, _, _, encoding = restore_dataset(
                input=wandb.config.input, cluster=wandb.config.cluster
            )

        else:
            raise Exception("Please pass an input .h5ad")

        X_train, y_train, X_test, y_test = data_utils.split_dataset(
            X_train,
            y_train,
            random_state=wandb.config.random_state,
            train_size=(1 - float(wandb.config.train_size)),
            estimator_type=ModelClass._estimator_type,
        )

        output = os.path.join(
            wandb.config.output, wandb.config.model_name, wandb.config.cluster, run.name
        )

        os.makedirs(output, exist_ok=True)
        assert os.path.exists(output) and os.path.isdir(output)

        TrainingConfig, HyperParameters = CONFIGS[wandb.config.model_name]
        hyperparameters = {
            hyperparam: getattr(wandb.config, hyperparam)
            for hyperparam in HyperParameters
        }

        config = AllConfig(
            training_config=TrainingConfig(**hyperparameters),
            random_state=wandb.config.random_state,
            cluster=wandb.config.cluster,
            output=output,
            model_name=wandb.config.model_name,
            device=device,
        )

        try:
            model, tolog = train(
                X_train.values,
                y_train.values,
                X_test.values,
                y_test.values,
                config=config,
                encoding=encoding,
            )
            model.save()
            model.save_results(config=config)
            wandb.log(tolog)
            # wandb.watch(model) # only valid with torch models

        except Exception as error:
            logging.error(traceback.print_exc())
            raise error


def sweep(
    input, model_name, cluster, output, config_file, project="uncategorized", sweeps=100
):

    with open(config_file, "r") as fh:
        config = json.load(fh)

    # NOTE
    # Documentation for the structure of a correct wandb configuration
    # is here: https://docs.wandb.ai/guides/sweeps/configuration
    config["parameters"]["model_name"] = {"values": [model_name]}
    config["parameters"]["output"] = {"values": [output]}
    config["parameters"]["cluster"] = {"values": [cluster]}
    config["parameters"]["input"] = {"values": [input]}
    config["name"] = get_sweep_name(cluster, model_name)

    project = config.pop("project", "uncategorized")
    sweep_id = wandb.sweep(config, project=project)
    wandb.agent(sweep_id, function=train_wandb, count=sweeps)
