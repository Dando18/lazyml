# std imports
import logging
from typing import Iterable, Optional, Tuple, Union

# tpl imports
from alive_progress import alive_it
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# local imports
from .dataset import Dataset
from .dimensionality_reduction import reduce_dimensionality
from .util import unlistify, without, expand_one_hot_columns, get_model_best


CLASSIFIER_MAP_ = {
    "Dummy": DummyClassifier,
    "Ridge": RidgeClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "SGD": SGDClassifier,
    "AdaBoost": AdaBoostClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "RandomForest": RandomForestClassifier,
    "GaussianProcess": GaussianProcessClassifier,
    "Perceptron": Perceptron,
    "GaussianNB": GaussianNB,
    "KNeighbors": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "SVM": SVC,
    "LinearSVM": LinearSVC,
    "XGB": XGBClassifier,
}


TUNING_PARAMETERS_MAP_ = {
    "Dummy": {"strategy": ["most_frequent", "prior", "stratified", "uniform"]},
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "fit_intercept": [True, False]},
    "DecisionTree": {"criterion": ["gini", "entropy", "log_loss"]},
    "SGD": {
        "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": ["l2", "l1", "elasticnet", None],
        "fit_intercept": [True, False],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [1e-2, 1e-1, 1.0],
    },
    "AdaBoost": {"n_estimators": [1, 10, 50, 100], "learning_rate": [0.01, 0.1, 1.0]},
    "GradientBoosting": {
        "n_estimators": [1, 10, 50, 100],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "RandomForest": {
        "criterion": ["gini", "entropy", "log_loss"],
        "n_estimators": [1, 10, 50, 100],
    },
    "GaussianProcess": None,
    "Perceptron": {
        "penalty": ["l2", "l1", "elasticnet"],
        "fit_intercept": [True, False],
    },
    "GaussianNB": None,
    "KNeighbors": {
        "n_neighbors": [1, 3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "MLP": {"hidden_layer_sizes": [(100,), (128, 64, 32), (256, 128, 64)]},
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 8],
    },
    "LinearSVM": {"C": [0.1, 1, 10], "fit_intercept": [True, False]},
    "XGB": {"n_estimators": [1, 10, 100, 200]},
}


def get_models_(models: Union[Iterable[Union[str, dict]], str]):
    if models == "all":
        return list(zip(CLASSIFIER_MAP_.values(), [None] * len(CLASSIFIER_MAP_)))
    elif models == "all-tuned":
        return [
            (CLASSIFIER_MAP_[name], TUNING_PARAMETERS_MAP_[name])
            for name in CLASSIFIER_MAP_.keys()
        ]
    else:
        all_models = []
        for model in models:
            model_name = model if isinstance(model, str) else model["name"]
            params = None if isinstance(model, str) else without(model, "name")
            all_models.append((CLASSIFIER_MAP_[model_name], params))
        return all_models


def train_classifiers(
    models: Union[Iterable[Union[str, dict]], str],
    train: Tuple[Union[np.ndarray, pd.DataFrame]],
    test: Tuple[Union[np.ndarray, pd.DataFrame]],
    metrics: Iterable[str] = ["accuracy"],
    models_partition: Tuple[int] = (0,1),
    dim_reduce_config: Optional[dict] = None,
    **kwargs,
):
    """Train each model on the dataset and return the best for each model."""
    (X_train, y_train), (X_test, y_test) = train, test

    results = []
    models = get_models_(models)
    models = np.array_split(np.array(models), models_partition[1])[models_partition[0]]
    for Clf, params in alive_it(models, title="Training"):
        try:
            scores = get_model_best(
                Clf, X_train, y_train, X_test, y_test, metrics, tune=params
            )
            results.extend(scores)
        except Exception as e:
            logging.debug(f"Classifier {Clf.__name__} failed.")
            logging.exception(e)
            continue

    return pd.DataFrame(results)
