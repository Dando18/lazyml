# std imports
import logging
import time
from typing import Iterable, Optional, Tuple, Union

# tpl imports
from alive_progress import alive_it
import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch,
    BisectingKMeans,
)
from sklearn.metrics import get_scorer

# local imports
from .dataset import Dataset
from .dimensionality_reduction import reduce_dimensionality
from .util import (
    unlistify,
    without,
    expand_one_hot_columns,
    mean_and_std_,
)


CLUSTERS_MAP_ = {
    "KMeans": KMeans,
    "AffinityPropagation": AffinityPropagation,
    "MeanShift": MeanShift,
    "Spectral": SpectralClustering,
    "Agglomerative": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "OPTICS": OPTICS,
    "Birch": Birch,
    "BisectingKMeans": BisectingKMeans,
}

TUNING_PARAMETERS_MAP_ = {}


def get_models_(models: Union[Iterable[Union[str, dict]], str]) -> list:
    if models == "all":
        return list(zip(CLUSTERS_MAP_.values(), [None] * len(CLUSTERS_MAP_)))
    elif models == "all-tuned":
        return [
            (CLUSTERS_MAP_[name], TUNING_PARAMETERS_MAP_[name])
            for name in CLUSTERS_MAP_.keys()
        ]
    else:
        all_models = []
        for model in models:
            model_name = model if isinstance(model, str) else model["name"]
            params = None if isinstance(model, str) else without(model, "name")
            all_models.append((CLUSTERS_MAP_[model_name], params))
        return all_models


def get_model_best(Cls, X_train, y_train, metrics, n_iter=5, **kwargs):
    tmp_results = {"time": []}
    tmp_results.update({m: [] for m in metrics})

    for _ in range(n_iter):
        clusterer = Cls(**kwargs)
        start = time.time()
        clusterer.fit(X_train)
        end = time.time()
        y_true, y_pred = y_train, clusterer.labels_

        tmp_results["time"].append(end - start)
        for metric in metrics:
            score = get_scorer(metric)._score_func(y_true, y_pred)
            tmp_results[metric].append(score)

    results = {"model": Cls.__name__}
    for k, v in tmp_results.items():
        results.update(mean_and_std_(v, k))

    return [results]


def train_clustering(
    models: Union[Iterable[Union[str, dict]], str],
    train: Tuple[Union[np.ndarray, pd.DataFrame]],
    test: Tuple[Union[np.ndarray, pd.DataFrame]],
    metrics: Iterable[str] = ["rand_score"],
    models_partition: Tuple[int] = (0, 1),
    dim_reduce_config: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Train clustering algorithms"""
    (X_train, y_train), (X_test, y_test) = train, test

    results = []
    models = get_models_(models)
    models = np.array_split(np.array(models), models_partition[1])[models_partition[0]]
    for Cls, params in alive_it(models, title="Training"):
        try:
            scores = get_model_best(Cls, X_train, y_train, metrics)
            results.extend(scores)
        except Exception as e:
            logging.debug(f"Clusterer {Cls.__name__} failed.")
            logging.exception(e)
            continue

    return pd.DataFrame(results)
