# std imports
import copy
import logging
import re
from typing import Any, Iterable, Optional, Union

# tpl imports
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, GridSearchCV


def without(d: dict, key: str, *vargs) -> dict:
    """Return a copy of dict without provided keys.

    Args:
        d: a dict
        key: key to remove from d
        vargs: other keys to remove from d

    Returns:
        a shallow copy of d with keys removed.
    """
    c = copy.copy(d)
    c.pop(key, None)
    for k in vargs:
        c.pop(k, None)
    return c


def unlistify(x: Iterable[Any]) -> Any:
    """if x is a 1 element list then return that element, otherwise return list.

    Args:
        x: an iterable

    Returns:
        The first element of x if it only has 1 element, otherwise x
    """
    if hasattr(x, "__len__") and len(x) == 1:
        return x[0]
    return x


def parse_columns(data: dict, dataset):
    """Handle `all-columns-except` option when providing columns as option. If given, then expands into the
    converse set of columns.
    If `columns-reg` is present, then column names are expanded from regex.

    Args:
        data: a dict that might have some column related parameters.
        dataset: dataset object
    """
    num_column_params = sum(
        1 for k in data.keys() if k in ["columns", "all-columns-except", "columns-regex"]
    )
    if num_column_params > 1:
        raise ValueError("Too many column parameters")

    if "columns" in data:
        pass  # do nothing in this case
    elif "all-columns-except" in data:
        data["columns"] = dataset.all_columns_except(data.pop("all-columns-except"))
    elif "columns-regex" in data:
        reg = re.compile(data.pop("columns-regex"))
        data["columns"] = list(filter(reg.match, dataset.train.columns))
    
    return data["columns"]


def expand_one_hot_columns(columns: Iterable[str], dataset) -> Iterable[str]:
    """If a column in columns was one-hot-encoded during preprocessing, then expand it into
    the corresponding columns.

    Args:
        columns: a list of column names
        dataset: dataset object

    Returns:
        The new list of column names after including one-hot columns.
    """
    new_cols = []
    for c in columns:
        if dataset.is_one_hot_column(c):
            new_cols.extend(dataset.get_one_hot_columns(c))
        else:
            new_cols.append(c)
    return new_cols


def mean_and_std_(arr: np.ndarray, title: str) -> dict:
    """Compute mean and std of a metric and return a dict with those values.

    Args:
        arr: several instances of some metric, np.ndarray
        title: name of metric

    Returns:
        A dict as { mean_title: mean(arr), std_title: std(arr) }
    """
    return {f"mean_{title}": np.mean(arr), f"std_{title}": np.std(arr)}


def format_params_(params: dict) -> dict:
    """Search through dict and replace True/False strings with bool values.

    Args:
        params: an arbitrary dict

    Returns:
        Bool string values in params replaced with actual bool values.
    """
    for key, value in params.items():
        if value in ["true", "True", "false", "False"]:
            params[key] = value.lower() == "true"
        elif hasattr(value, "__iter__"):
            for idx, v in enumerate(value):
                if v in ["true", "True", "false", "False"]:
                    params[key][idx] = v.lower() == "true"


def get_model_best(
    Model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.DataFrame],
    metrics: Iterable[str],
    tune: Optional[dict] = None,
    seed: int = 42,
    **params,
) -> list:
    """Find an approximate best score from Model on X and y.

    Args:
        Model: class of learning model. Must implement fit and score functions like sklearn estimators.
        X: training dataset
        y: training output target
        X_test: testing dataset
        y_test: testing output target
        metrics: metrics to compute on each model; must be sklearn metric strings.
        tune: If not None, then tune the model over these hyperparameters.
        seed: random seed for training.
        params: params used to instantiate model.

    Returns:
        A list of dicts. Each dict contains the best result for each metric. Can be used to create DataFrame.
    """
    logging.debug(f"Training model: {Model.__name__}")
    clf = Model(**params)
    results = []

    if tune:
        format_params_(tune)
        search = GridSearchCV(clf, tune, scoring=metrics, refit=False)
        search.fit(X, y)

        df = pd.DataFrame(search.cv_results_)
        # for each metric choose first row where rank_test_metric is 1
        for metric in metrics:
            best = df[df[f"rank_test_{metric}"] == 1].iloc[0]
            clf.set_params(**best["params"])
            clf.fit(X_test, y_test)
            y_true, y_pred = y_test, clf.predict(X_test)

            metric_results = {"model": Model.__name__}
            metric_results.update(
                {
                    "mean_time": best["mean_fit_time"],
                    "std_time": best["std_fit_time"],
                    "params": best["params"],
                }
            )
            metric_results.update(
                {f"mean_{m}": best[f"mean_test_{m}"] for m in metrics}
            )
            metric_results.update({f"std_{m}": best[f"std_test_{m}"] for m in metrics})
            metric_results.update(
                {
                    f"test_{m}": get_scorer(m)._score_func(y_true, y_pred)
                    for m in metrics
                }
            )
            results.append(metric_results)
    else:
        cv_results = cross_validate(clf, X, y, scoring=metrics)
        clf.fit(X, y)
        y_true, y_pred = y_test, clf.predict(X_test)

        tmp = {"model": Model.__name__}
        tmp.update(mean_and_std_(cv_results["fit_time"], "time"))
        for metric in metrics:
            tmp.update(mean_and_std_(cv_results[f"test_{metric}"], metric))

            score = get_scorer(metric)._score_func(y_true, y_pred)
            tmp[f"test_{metric}"] = score

        results.append(tmp)

    return results
