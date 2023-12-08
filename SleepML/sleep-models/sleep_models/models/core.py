import logging

from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_validate
from sleep_models.models import MODELS


def train(X_train, y_train, X_test, y_test, config, encoding):

    ModelClass = MODELS[config.model_name]
    model = ModelClass.new_model(X_train, y_train, config, encoding=encoding)
    model.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    loss = model.get_loss(X_train, y_train)
    test_loss = model.get_loss(X_test, y_test)

    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Computing Pearson's r")
    corr, p = pearsonr(y_train.argmax(axis=1), y_pred.argmax(axis=1))
    corr_test, p_test = pearsonr(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))

    print("Computing performance metrics")
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
        "actual_epochs": model.epochs
        # "mean_squared_error": msqe,
    }

    print(tolog)
    return model, tolog
