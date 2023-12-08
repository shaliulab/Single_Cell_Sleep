# To add a new model:
# Keep calm and...
#
#
# 1. Create a class with the scikit learn API
# (either from scratch or subclassing)
#
# 2. Define the class' _metric = "ACC"
#
# 3. Add that class to the MODELS list at the end
#
#
#

import logging
from abc import ABC, abstractmethod
import os.path
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.utils.estimator_checks import check_estimator
from sleep_models.preprocessing import make_confusion_table
from sleep_models.models.utils.config import isnamedtupleinstance
from sleep_models.plotting import plot_confusion_table
from sleep_models.preprocessing import make_confusion_long_to_square
from sleep_models.models.variables import ModelProperties

logger = logging.getLogger(__name__)


class SleepModel(ABC):

    _estimator_type = None
    uses_test_in_train = False

    def __init__(self, name, target, output=".", random_state=1000):
        self.name = name
        self.output = output
        self.random_state = random_state
        self._target = target
        super().__init__()

    @classmethod
    @abstractmethod
    def new_model(cls, config, X_train=None, y_train=None, encoding=None):
        raise NotImplementedError

    @property
    def epochs(self):
        return 0

    @classmethod
    def model_properties(cls):
        return ModelProperties(
            encoding=cls._encoding, estimator_type=cls._estimator_type
        )

    def benchmark(self, X, y):

        prediction = self.predict(X)

        accuracy = accuracy_score(y, prediction)
        loss = log_loss(y, prediction)

        prediction_code = np.array(prediction).argmax(1)
        right = prediction_code == y.argmax(1)

        return accuracy, loss, prediction, right

    def fit(self, X, y, *args, X_test=None, y_test=None, **kwargs):
        return super(SleepModel, self).fit(X, y)

    def get_loss(self, X, y):
        """
        Returns the loss of the model for this dataset

        Arguments:
            X (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories
        Returns:
            loss (float): Number from 0 to Infinity quantifying how wrong the model is
                loss of 0 means the model asssigns all probability to the truth i.e 0 to wrong options
                A non zero loss is compatible with 100% accuracy, because predictions can be made
                correctly while still assigning some probability to the wrong classes
        """

        _, loss, _, _ = self.benchmark(X, y)
        return loss

    def get_metric(self, X, y):
        return getattr(self, f"_get_{self._metric.lower()}")(X, y)

    def compute_metric(self, X, y):
        warnings.warn("Please use get_metric")
        return self.get_metric(X, y)

    def _get_accuracy(self, X, y):
        return self.score(X, y)

    def _get_rmse(self, X, y):
        """
        Compute Root Mean Squared Error for this dataset

        Arguments:
            X (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories        
        """

        y_pred = self.predict(X).argmax(1)
        y_flat = y.argmax(1)
        return np.sqrt(mean_squared_error(y_pred, y_flat))

    def get_confusion_table(self, truth, predictions):
        """
        Compute confusion table for this dataset

        Arguments:
            X (np.ndarray): Shape nxm where n = number of samples (single cells) and m number of features
            y (np.ndarray): Shape nxc where n = number of samples (single cells) and c number of categories        
        """
        order = list(self._label_code.values())

        confusion_table = make_confusion_long_to_square(
            make_confusion_table(self, truth, predictions), order=order
        )

        return confusion_table

    def save(self):
        path = os.path.join(self.output, f"{self.name}.pickle")
        print(f"Saving model to {path}")
        with open(path, "wb") as filehandle:
            pickle.dump(self, filehandle)

    def save_metrics(self):
        return

    def save_results(self, suffix=None, **kwargs):

        self.save_metrics()

        for key, value in kwargs.items():

            components = [self.name, key, suffix]
            components = [c for c in components if c is not None]

            base_filename = "_".join(components)

            # confusion table
            if isinstance(value, pd.DataFrame):
                value.to_csv(os.path.join(self.output, base_filename + ".csv"))

                if key == "confusion_table":
                    confusion_table = value
                    print(confusion_table)

                    plot_confusion_table(
                        confusion_table,
                        os.path.join(self.output, f"{base_filename}.png"),
                    )

            # config
            elif isinstance(value, tuple) and isnamedtupleinstance(value):
                data = {k: getattr(value, k) for k in value._fields}
                with open(
                    os.path.join(self.output, base_filename + ".yml"), "w"
                ) as filehandle:
                    yaml.dump(data, filehandle)


class EBM(
    SleepModel, ExplainableBoostingClassifier,
):

    _estimator_type = "classifier"
    _encoding = "ONE_HOT"
    _metric = "accuracy"
    uses_test_in_train = True

    def __init__(
        self,
        name,
        target,
        output=".",
        random_state=1000,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.01,
        validation_size=0.15,
        min_samples_leaf=2,
        max_leaves=3,
        max_rounds=5000,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
    ):
        ExplainableBoostingClassifier.__init__(
            self,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            validation_size=validation_size,
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
        )

        super(EBM, self).__init__(
            name=name, target=target, output=output, random_state=random_state
        )
        self._ncols = None

    @classmethod
    def new_model(cls, config, X_train=None, y_train=None, encoding=None):
        model = cls(
            name=config.cluster,
            output=config.output,
            random_state=config.random_state,
            target=config.target,
        )
        model._label_code = encoding
        return model

    def fit(self, X, y, *args, X_test=None, y_test=None, **kwargs):
        self._ncols = y.shape[1]
        y_truth = y.argmax(1)
        result = super(EBM, self).fit(X, y_truth)

        y_pred_test = self.predict(X_test)

        confusion_table = self.get_confusion_table(
            y_test.argmax(1), y_pred_test.argmax(1)
        )
        self.save()
        self.save_results(confusion_table=confusion_table)
        return result

    def predict(self, X):
        y_pred = super().predict(X)
        ph = np.zeros((y_pred.shape[0], self._ncols))

        for i in range(y_pred.shape[0]):
            ph[i, y_pred[i]] = 1

        return ph

    def benchmark(self, X, y):

        y2 = y.argmax(1)
        prediction = self.predict(X)
        accuracy = accuracy_score(y, prediction)
        loss = log_loss(y, prediction)

        prediction_code = prediction
        right = prediction_code == y

        return accuracy, loss, prediction, right

    def save(self):
        path = os.path.join(self.output, f"{self.name}.joblib")
        print(f"Saving model to {path}")
        with open(path, "wb") as filehandle:
            joblib.dump(self, filehandle)


class RF(SleepModel, RandomForestClassifier):

    _estimator_type = "classifier"
    _encoding = "ONE_HOT"
    _metric = "accuracy"

    @classmethod
    def new_model(cls, config, X_train=None, y_train=None, encoding=None):
        model = cls(
            name=config.cluster,
            output=config.output,
            random_state=config.random_state,
            target=config.target,
        )
        model._label_code = encoding
        return model


class KNN(
    SleepModel, KNeighborsClassifier,
):
    _estimator_type = "classifier"
    _encoding = "ONE_HOT"
    _metric = "accuracy"

    @classmethod
    def new_model(cls, config, X_train=None, y_train=None, encoding=None):
        model = cls(
            name=config.cluster,
            n_neighbors=config.training_config.n_neighbors,
            weights=config.training_config.weights,
            leaf_size=config.training_config.leaf_size,
            p=config.training_config.p,
            metric=config.training_config.metric,
            random_state=config.random_state,
            target=config.target,
        )

        model._label_code = encoding
        return model


class MLP(
    SleepModel, MLPRegressor,
):

    _estimator_type = "classifier"
    _encoding = "ONE_HOT"
    _metric = "accuracy"

    @classmethod
    def new_model(cls, config, X_train=None, y_train=None, encoding=None):
        model = cls(
            name=config.cluster,
            hidden_layer_sizes=config.training_config.n_neurons,
            activation=config.training_config.activation,
            solver=config.training_config.solver,
            alpha=config.training_config.alpha,
            batch_size=config.training_config.batch_size,
            learning_rate=config.training_confignfig.learning_rate,
            learning_rate_init=config.training_config.learning_rate_init,
            random_state=config.random_state,
            target=config.target,
        )

        model._label_code = encoding
        return model
