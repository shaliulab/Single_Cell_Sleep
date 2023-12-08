import logging

logger = logging.getLogger(__name__)
import os.path


def fit_model(model, data):

    X_train, y_train, X_test, y_test = data

    logger.info("Training model")

    if model.uses_test_in_train:
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    else:
        model.fit(X_train, y_train)


def train_model(model, data):
    """
    Arguments:

        * model (sleep_models.models.models.SleepModel): An instance of one of the models in sleep_models
        * data (tuple of np.ndarray): tuple of length 4 with the features and labels of a train and a test set
    """

    X_train, y_train, X_test, y_test = data

    fit_model(model, (X_train, y_train, X_test, y_test))

    logger.info("Backing up model")
    model.save()

    # y_pred_train = model.predict(X_train)
    # y_pred = model.predict(X_test)

    # fig, axes = model.plot_model_performance(y_pred_train, y_pred)
    # fig.savefig(os.path.join(model._output, f"{model._cluster}-scatter.png"))

    print(f"{model._metric} on train set: {model.get_metric(X_train, y_train)}")
    print(f"{model._metric} on test set: {model.get_metric(X_test, y_test)}")
    return 0
