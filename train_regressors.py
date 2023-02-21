# std imports
import logging
from typing import Iterable, Optional, Union

# tpl imports
from alive_progress import alive_it
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, PassiveAggressiveRegressor, \
                                 PoissonRegressor, Lasso
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor


# local imports
from dataset import Dataset
from dimensionality_reduction import reduce_dimensionality
from util import unlistify, without, expand_one_hot_columns

# GaussianProcessRegressor is excluded because it always crashes for me
REGRESSORS_MAP_ = {
    'Dummy': DummyRegressor,
    'Linear': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'SGD': SGDRegressor,
    'PassiveAggressive': PassiveAggressiveRegressor,
    'Poisson': PoissonRegressor,
    'KNeighbors': KNeighborsRegressor,
    'AdaBoost': AdaBoostRegressor,
    'GradientBoosting': GradientBoostingRegressor,
    'RandomForest': RandomForestRegressor,
    'DecisionTree': DecisionTreeRegressor,
    'MLP': MLPRegressor,
    'SVM': SVR,
    'LinearSVM': LinearSVR
}


TUNING_PARAMETERS_MAP_ = {
    'Dummy': {'strategy': ["mean", "median"]}
}



def mean_and_std_(arr : np.ndarray, title : str):
    return { f'mean_{title}': np.mean(arr), f'std_{title}': np.std(arr) }


def format_params(params : dict) -> dict:
    for key, value in params.items():
        if value in ['true', 'True', 'false', 'False']:
            params[key] = (value.lower() == 'true')
        elif hasattr(value, '__iter__'):
            for idx, v in enumerate(value):
                if v in ['true', 'True', 'false', 'False']:
                    params[key][idx] = (v.lower() == 'true')


def get_regressor_best_(
    Regressor,
    X : Union[np.ndarray, pd.DataFrame],
    y : Union[np.ndarray, pd.DataFrame],
    X_test : Union[np.ndarray, pd.DataFrame],
    y_test : Union[np.ndarray, pd.DataFrame],
    metrics : Iterable[str],
    tune : Optional[dict] = None,
    seed : int = 42,
    **params
):
    ''' Find an approximate best score from reg on X and y.
    '''
    logging.debug(f'Training classifier: {Regressor.__name__}')
    reg = Regressor(**params)
    results = []

    if tune:
        format_params(tune)
        search = GridSearchCV(reg, tune, scoring=metrics, refit=False)
        search.fit(X, y)

        df = pd.DataFrame(search.cv_results_)
        # for each metric choose first row where rank_test_metric is 1
        for metric in metrics:
            best = df[df[f'rank_test_{metric}'] == 1].iloc[0]
            reg.set_params(**best['params'])
            reg.fit(X_test, y_test)
            y_true, y_pred = y_test, reg.predict(X_test)

            metric_results = {'model': Regressor.__name__}
            metric_results.update({'mean_time': best['mean_fit_time'],  'std_time': best['std_fit_time'], 
                'params': best['params']})
            metric_results.update({f'mean_{m}': best[f'mean_test_{m}'] for m in metrics})
            metric_results.update({f'std_{m}': best[f'std_test_{m}'] for m in metrics})
            metric_results.update({f'test_{m}' : get_scorer(m)._score_func(y_true, y_pred) for m in metrics})
            results.append(metric_results)
    else:
        cv_results = cross_validate(reg, X, y, scoring=metrics)
        reg.fit(X, y)
        y_true, y_pred = y_test, reg.predict(X_test)

        tmp = {'model': Regressor.__name__}
        tmp.update(mean_and_std_(cv_results['fit_time'], 'time'))
        for metric in metrics:
            tmp.update(mean_and_std_(cv_results[f'test_{metric}'], metric))

            score = get_scorer(metric)._score_func(y_true, y_pred)
            tmp[f'test_{metric}'] = score

        results.append(tmp)
    
    return results



def get_models_(models : Union[Iterable[Union[str,dict]], str]):
    if models == 'all':
        return list(zip(REGRESSORS_MAP_.values(), [None]*len(REGRESSORS_MAP_)))
    elif models == 'all-tuned':
        return [(REGRESSORS_MAP_[name], TUNING_PARAMETERS_MAP_[name]) for name in REGRESSORS_MAP_.keys()]
    else:
        all_models = []
        for model in models:
            model_name = model if isinstance(model, str) else model['name']
            params = None if isinstance(model, str) else without(model, 'name')
            all_models.append((REGRESSORS_MAP_[model_name], params))
        return all_models


def train_regressors(
    dataset : Dataset, 
    models : Union[Iterable[Union[str,dict]], str], 
    X_columns : Optional[Union[str, Iterable[str]]] = None, 
    y_columns : Optional[Union[str, Iterable[str]]] = None, 
    metrics : Iterable[str] = ['neg_mean_absolute_error', 'r2'],
    dim_reduce_config : Optional[dict] = None,
    **kwargs
) -> pd.DataFrame:
    ''' Train each model on the dataset and return the best for each model.
    '''
    if X_columns is None and y_columns is None:
        raise ValueError('Must provide at least 1 of \'X_columns\' or \'y_columns\'.')
    
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
                                dim_reduce_config['name'], 
                                X_train, X_test, 
                                **without(dim_reduce_config, 'name')
                            )

    results = []
    models = get_models_(models)
    for Reg, params in alive_it(models, title='Training'):
        try:
            scores = get_regressor_best_(Reg, X_train, y_train, X_test, y_test, metrics, tune=params)
            results.extend(scores)
        except Exception as e:
            logging.debug(f'Regressor {Reg.__name__} failed.')
            logging.exception(e)
            continue

    return pd.DataFrame(results)
