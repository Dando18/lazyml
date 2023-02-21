""" Functions for training regressors.
    author: Daniel Nichols
    date: February, 2022
"""

# std imports
import logging
from typing import Iterable, Optional, Union

# tpl imports
from alive_progress import alive_it
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    SGDRegressor,
    PassiveAggressiveRegressor,
    PoissonRegressor,
    Lasso,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

# local imports
from .dataset import Dataset
from .dimensionality_reduction import reduce_dimensionality
from .util import unlistify, without, expand_one_hot_columns, get_model_best


# GaussianProcessRegressor is excluded because it always crashes for me
REGRESSORS_MAP_ = {
    "Dummy": DummyRegressor,
    "Linear": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "SGD": SGDRegressor,
    "PassiveAggressive": PassiveAggressiveRegressor,
    "Poisson": PoissonRegressor,
    "KNeighbors": KNeighborsRegressor,
    "AdaBoost": AdaBoostRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "RandomForest": RandomForestRegressor,
    "DecisionTree": DecisionTreeRegressor,
    "MLP": MLPRegressor,
    "SVM": SVR,
    "LinearSVM": LinearSVR,
}


TUNING_PARAMETERS_MAP_ = {
    "Dummy": {"strategy": ["mean", "median"]},
    "Linear": {"fit_intercept": [True, False]},
    "Ridge": {"alpha": [0.01, 0.1, 1, 10], "fit_intercept": [True, False]},
    "Lasso": {"alpha": [0.01, 0.1, 1, 10], "fit_intercept": [True, False]},
    "SGD": {
        "loss": ["huber", "squared_error"],
        "penalty": ["l2", "l1", "elasticnet", None],
        "fit_intercept": [True, False],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [1e-2, 1e-1, 1.0],
    },
    "PassiveAggressive": {"C": [0.1, 1, 10], "fit_intercept": [True, False]},
    "Poisson": {"alpha": [0.01, 0.1, 1, 10], "fit_intercept": [True, False]},
    "KNeighbors": {
        "n_neighbors": [1, 3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "AdaBoost": {"n_estimators": [1, 10, 50, 100], "learning_rate": [0.01, 0.1, 1.0]},
    "GradientBoosting": {
        "n_estimators": [1, 10, 50, 100],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "RandomForest": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        "n_estimators": [1, 10, 50, 100],
    },
    "DecisionTree": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
    "MLP": {"hidden_layer_sizes": [(100,), (128, 64, 32), (256, 128, 64)]},
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 8],
    },
    "LinearSVM": {"C": [0.1, 1, 10], "fit_intercept": [True, False]},
}


def get_models_(models: Union[Iterable[Union[str, dict]], str]) -> list:
    if models == "all":
        return list(zip(REGRESSORS_MAP_.values(), [None] * len(REGRESSORS_MAP_)))
    elif models == "all-tuned":
        return [
            (REGRESSORS_MAP_[name], TUNING_PARAMETERS_MAP_[name])
            for name in REGRESSORS_MAP_.keys()
        ]
    else:
        all_models = []
        for model in models:
            model_name = model if isinstance(model, str) else model["name"]
            params = None if isinstance(model, str) else without(model, "name")
            all_models.append((REGRESSORS_MAP_[model_name], params))
        return all_models


def train_regressors(
    dataset: Dataset,
    models: Union[Iterable[Union[str, dict]], str],
    X_columns: Optional[Union[str, Iterable[str]]] = None,
    y_columns: Optional[Union[str, Iterable[str]]] = None,
    metrics: Iterable[str] = ["neg_mean_absolute_error", "r2"],
    dim_reduce_config: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Train each model on the dataset and return the best for each model."""
    if X_columns is None and y_columns is None:
        raise ValueError("Must provide at least 1 of 'X_columns' or 'y_columns'.")

    if X_columns is None:
        X_columns = dataset.all_columns_except(y_columns)
    if y_columns is None:
        y_columns = dataset.all_columns_except(X_columns)

    X_columns = expand_one_hot_columns(X_columns, dataset)
    y_columns = expand_one_hot_columns(y_columns, dataset)

    y_columns = unlistify(y_columns)
    X_train, y_train = dataset.train[X_columns], dataset.train[y_columns]
    X_test, y_test = dataset.test[X_columns], dataset.test[y_columns]

    if dim_reduce_config:
        X_train, X_test = reduce_dimensionality(
            dim_reduce_config["name"],
            X_train,
            X_test,
            **without(dim_reduce_config, "name"),
        )

    results = []
    models = get_models_(models)
    for Reg, params in alive_it(models, title="Training"):
        try:
            scores = get_model_best(
                Reg, X_train, y_train, X_test, y_test, metrics, tune=params
            )
            results.extend(scores)
        except Exception as e:
            logging.debug(f"Regressor {Reg.__name__} failed.")
            logging.exception(e)
            continue

    return pd.DataFrame(results)
