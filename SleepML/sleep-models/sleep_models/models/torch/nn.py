import os.path
import itertools
import warnings
import pickle
import logging
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from torch.utils.data import DataLoader

from sleep_models.models.torch.train import train_model
from sleep_models.models.torch.test import test
from sleep_models.models.torch.data import SingleCellDataSet
import sleep_models.models.utils.metrics as metrics_utils
import sleep_models.models.utils.config as config_utils
from sleep_models.models.models import SleepModel
from sleep_models.models.torch.tools import EarlyStopping


class SkLearnAPI:

    """
    Provide a scikit-learn-like API to ease integration with scikit-learn tools
    """

    def fit(self, X, y, X_test=None, y_test=None):

        """
        Train the model so it learns a mapping between X and y
        Optionally provide a test set to monitor the loss during training

        Arguments:
            X (np.array): nxm array of n samples and m features
            y (np.array): nxc array of n samples and c classes after encoding
            X_test (np.array): nxm array of n samples and m features.
            Weights are not updated based on this dataset
            y_test (np.array): nxc array of n samples and c classes after encoding
            Weights are not updated based on this dataset

        Returns:
            None
        """

        X_train = X
        y_train = y

        if X_test is None:
            X_test = X_train.copy()
            warnings.warn(f"{self} will use X_train as X_test")
        if y_test is None:
            y_test = y_train.copy()
            warnings.warn(f"{self} will use y_train as y_test")

        metrics_utils.calibrate_accuracy(y_test)
        training_data = SingleCellDataSet(X_train, y_train, device=self.device)
        test_data = SingleCellDataSet(X_test, y_test, device=self.device)
        data = {"training": training_data, "test": test_data}

        # the test dataset is only used to stop when the loss stops decreasing
        # BUT IT DOES NOT
        # drive the gradient optimization
        epochs, metrics = train_model(self, data)

        self._metrics = metrics
        self._epochs = epochs
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict targets of the input dataset using a trained model

        Arguments:
            X (np.array): nxm array of n samples and m features
        
        Returns:
            prediction (np.array) nxc array of encoded predictions

        """

        check_is_fitted(self, "is_fitted_")
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        logits = self(X)
        prediction = self.prob_layer(logits)
        return prediction.cpu().numpy()

    def benchmark(self, X, y):
        """
        Pass the features through the model and compare the prediction to the ground truth

        Arguments:
            X (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories

        Returns:

            accuracy (float): Number from 0 to 1 quantifying the accuracy of the model
                1.0 means all predictions match the ground truth, and 0.0 means none match
            loss (float): Number from 0 to Infinity quantifying how wrong the model is
                loss of 0 means the model asssigns all probability to the truth i.e 0 to wrong options
                A non zero loss is compatible with 100% accuracy, because predictions can be made
                correctly while still assigning some probability to the wrong classes
            prediction (list): List of int with length equal to the number of samples,
                storing on the ith element the code of the ith ground truth label
            right (list): List of bool with length equal to the number of samples,
                storing on the ith element whether the ith sample is predicted right or wrong
        """

        if not isinstance(X, np.ndarray):
            logging.warning(
                f"Please pass a np.ndarray in the X slot instead of {type(X)}"
            )
            X = X.values
        if not isinstance(y, np.ndarray):
            logging.warning(
                f"Please pass a np.ndarray in the y slot instead of {type(y)}"
            )
            y = y.values

        device = self.device
        test_data = SingleCellDataSet(X, y, device=device)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        accuracy, loss, prediction, right = test(model=self, dataloader=test_dataloader)
        return accuracy, loss, prediction, right

    def score(self, X, y):
        """
        Returns the score (accuracy) of the model for this dataset
    
        Arguments:
            X (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories
        Returns:
            accuracy (float): Number from 0 to 1 quantifying the accuracy of the model
                1.0 means all predictions match the ground truth, and 0.0 means none match
        """

        accuracy, _, _, _ = self.benchmark(X, y)
        return accuracy


class NeuralNetwork(SkLearnAPI, SleepModel, nn.Module):

    HIDDEN_NEURONS = 200
    _encoding = "ONE_HOT"
    _estimator_type = "classifier"
    _metric = "accuracy"
    uses_test_in_train = True  # only to decide whether to stop or not

    def __init__(
        self, n_features, n_classes, n_neurons, dropouts, config, *args, **kwargs
    ):
        """
        Initialize a Neural Network model

        n_features (int): Number of features (genes), used as input
        n_classes (int): Number of classes (states, treatments, conditions, etc) to be predicted
        n_neurons (list): Number of hidden neurons on each hidden layer. Length of this list is equal to the number of hidden layers
        dropouts (list): Dropout probability during training before each layer. Dont pass anything to set no dropouts
        There should be up to number of hidden layers + 1 dropouts. If less are passed, the remaining ones are assumed to be p=0 (i.e. no dropout)
        name (str): Name of model, useful to distinguish it from other models and to create its output
        output (str): Output folder of the model
        """
        super(NeuralNetwork, self).__init__(*args, **kwargs)
        self._n_features = n_features
        self._n_classes = n_classes
        self._n_neurons = n_neurons
        self._all_neurons = [n_features] + n_neurons + [n_classes]
        self._dropouts = dropouts
        self.l2 = config.l2
        self.learning_rate = config.learning_rate
        self.max_iter = config.epochs
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer

        if config.early_stopping:
            early_stopping = EarlyStopping(
                patience=config.patience, verbose=True, should_decrease=False
            )
        else:
            # if no early stopping is passed, use "infinitely" late stopping
            early_stopping = EarlyStopping(
                patience=99999999999, verbose=True, should_decrease=False
            )

        self.early_stopping = early_stopping
        self.best_metrics = {}
        self._epochs = 0
        self.is_fitted = False

        layers = self.make_layers(self._all_neurons, dropouts)
        self.linear_relu_stack = layers

        print("Model: ")
        print(self.linear_relu_stack)

    @property
    def n_neurons(self):
        return self._n_neurons

    @property
    def dropouts(self):
        return self.dropouts

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_features(self):
        return self._n_features

    @property
    def cluster(self):
        """
        Name of the dataset that the model was built and trained for
        """
        logging.warning("cluster attribute is deprecated. Please use .name instead")
        return self.name

    @classmethod
    def new_model(cls, config, X_train, y_train, encoding):
        """
        Produce a new instance of a Neural Network

        Arguments:
            X_train (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y_train (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories
            config (sleep_models.variables.AllConfig): Named tuple with attributes training_config, cluster, output, device, model_name
            encoding (dict): Mapping of codes (keys) to labels (values)
        """

        n_features = X_train.shape[1]
        if len(y_train.shape) > 1:
            n_classes = y_train.shape[1]
        else:
            n_classes = 1

        dropouts = config.training_config.dropouts
        n_neurons = config.training_config.n_neurons

        if len(dropouts) != (len(n_neurons) + 1):
            logging.warning(
                "Please specify number_of_hidden_layers+1 dropouts. I will assume dropout is always 0 then (i.e. no dropout)"
            )
            while len(dropouts) != (len(n_neurons) + 1):
                dropouts.append(0)

        print(f"Number of features: {n_features}")
        print(f"Number of classes: {n_classes}")

        model = cls(
            name=config.cluster,
            n_features=n_features,
            n_classes=n_classes,
            n_neurons=n_neurons,
            dropouts=dropouts,
            output=config.output,
            config=config.training_config,
            random_state=config.random_state,
            target=config.target,
        ).to(config.device)

        labels = y_train.argmax(1)

        model.loss_function = config_utils.get_loss_function(
            loss_function=config.training_config.loss_function, labels=labels
        ).to(model.device)

        model._label_code = encoding
        return model

    @property
    def epochs(self):
        """
        Number of epochs that the model has been trained on
        """
        return self._epochs

    @property
    def device(self):
        """
        Return "cuda" if running on gpu and "cpu" otherwise
        Taken from https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/11
        """
        if next(self.parameters()).device.type == "cuda":
            return "cuda"
        else:
            return "cpu"

    @classmethod
    def make_layer(cls, n_neurons, dropout):
        """
        Add an extra layer of hidden neurons to the model with
            rectified linear unit to provide non linearity
            optional dropout

        Arguments:
        
            n_neurons (int) Number of hidden neurons in the new layer
            dropout (float): Probability of neuron inactivation from 0 to 1
    
        Returns:
            layer (list): The optional dropout, the hidden neurons and the non linearity
        """

        if dropout != 0:
            layer = [nn.Dropout(dropout)]
        else:
            layer = []

        layer += [nn.Linear(*n_neurons), nn.ReLU()]

        return layer

    def make_layers(self, n_neurons, dropouts):

        layers = []
        for i in range(0, len(n_neurons) - 1):
            layer = self.make_layer(
                n_neurons=n_neurons[(i) : (i + 2)], dropout=dropouts[i]
            )

            if i == (len(n_neurons) - 2):
                layer = layer[:-1]
            layers.append(layer)

        DIM = 1
        self._prob_layer = nn.Softmax(dim=DIM)
        return nn.Sequential(*list(itertools.chain(*layers)))

    def prob_layer(self, logits):
        """
        Compute the probability of each class,
        given the raw output of a model
        """

        # dim=1 means every slice adds up to 1
        # where one slice is one row (one single cell)
        prob = self._prob_layer(logits).detach()
        return prob

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def save(self):
        path = os.path.join(self.output, f"{self.name}.pth")
        print(f"Saving model to {path}")
        torch.save(self, path)

    def save_metrics(self):
        with open(
            os.path.join(self.output, f"{self.name}_metrics.yaml"), "w"
        ) as filehandle:
            for metric in self.best_metrics:
                filehandle.write(f"{metric}: {str(self.best_metrics[metric])}\n")

    def dump(self, model_name):
        with open(os.path.join(self.output, model_name), "wb") as fh:
            pickle.dump(self, fh)

        self.save()
        super().dump()

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        return model
