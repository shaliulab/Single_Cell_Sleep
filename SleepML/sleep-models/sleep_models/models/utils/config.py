import logging
import os.path

import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import yaml

from sleep_models.models.torch.tools import EarlyStopping
from sleep_models.constants import DEFAULT_DICT
from sleep_models.constants import TRAINING_PARAMS
from sleep_models.models.variables import CONFIGS


def load_config():

    output = DEFAULT_DICT.copy()
    if os.path.exists(TRAINING_PARAMS):
        with open(TRAINING_PARAMS, "r") as filehandle:
            config = yaml.load(filehandle, yaml.SafeLoader)

        output.update(config)
    return output


def get_loss_function(loss_function, labels):
    """
    Initialize a CrossEntropyLoss with mean reduction and class weights to coutnerbalance overrepresented classes

    Arguments:
        loss_function (str): Name of a loss function defined in torch.nn that takes weight and reduction arguments
        labels (list): List of length equal to N where element i encodes the class or label of sample i
    """

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Initialize the loss function
    try:
        loss_function = getattr(nn, loss_function)(
            weight=class_weights, reduction="mean"
        )
    except Exception as error:
        logging.error("Could not initialize loss function" f" {error}")
    return loss_function


def setup_config(model, **kwargs):

    """
    Given a model, a training configuration and labels,
    initialize a Config tuple containing:

    * learning rate
    * batch size
    * L2 regularization coefficient
    * Number of epochs
    * Early stopping
    * Optimizer to be used (needs model object)
    * Loss function (needs labels object)

    Arguments:
        kwargs (dict): Extra values to be saved in the config file
    """

    config = load_config()

    # loss_function = get_loss_function(loss_function=loss_function, labels=labels).to(device)
    early_stopping = EarlyStopping(
        patience=config["patience"], verbose=True, should_decrease=False
    )

    TrainingConfig, HyperParameters = CONFIGS[model]
    hyperparameters = {hyperparam: config[hyperparam] for hyperparam in HyperParameters}
    hyperparameters.update(kwargs)
    config = TrainingConfig(**hyperparameters)
    return config


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)
