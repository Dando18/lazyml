# std imports
import logging
from typing import Iterable, Optional, Union

# tpl imports
from alive_progress import alive_it
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


# local imports
from dataset import Dataset
from util import unlistify, without


CLASSIFIER_MAP_ = {
    'Dummy': DummyClassifier,
    'Ridge': RidgeClassifier,
    'DecisionTree': DecisionTreeClassifier,
    'SGD': SGDClassifier,
    'AdaBoost': AdaBoostClassifier,
    'GradientBoosting': GradientBoostingClassifier,
    'RandomForest': RandomForestClassifier,
    'GaussianProcess': GaussianProcessClassifier,
    'Perceptron': Perceptron,
    'GaussianNB': GaussianNB,
    'KNeighbors': KNeighborsClassifier,
    'MLP': MLPClassifier,
    'SVM': SVC,
    'LinearSVM': LinearSVC
}


TUNING_PARAMETERS_MAP_ = {
    'Dummy': {'strategy': ["most_frequent", "prior", "stratified", "uniform"]},
    'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'fit_intercept': [True, False]},
    'DecisionTree': {'criterion': ["gini", "entropy", "log_loss"]},
    'SGD': {'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 
            'penalty': ['l2', 'l1', 'elasticnet', None], 'fit_intercept': [True, False],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0': [1e-2, 1e-1, 1.0]},
    'AdaBoost': {'n_estimators': [1, 10, 50, 100], 'learning_rate': [0.01, 0.1, 1.0]},
    'GradientBoosting': {'n_estimators': [1, 10, 50, 100], 'learning_rate': [0.01, 0.1, 1.0]},
    'RandomForest': {'criterion': ["gini", "entropy", "log_loss"], 'n_estimators': [1, 10, 50, 100]},
    'GaussianProcess': None,
    'Perceptron': {'penalty': ['l2','l1','elasticnet'], 'fit_intercept': [True, False]},
    'GaussianNB': None,
    'KNeighbors': {'n_neighbors': [1, 3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1,2]},
    'MLP': {'hidden_layer_sizes': [(100,), (128, 64, 32), (256, 128, 64)]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 8]},
    'LinearSVM': {'C': [0.1, 1, 10], 'fit_intercept': [True, False]}
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


def get_classifier_best_(
    Classifier,
    X : Union[np.ndarray, pd.DataFrame],
    y : Union[np.ndarray, pd.DataFrame],
    X_test : Union[np.ndarray, pd.DataFrame],
    y_test : Union[np.ndarray, pd.DataFrame],
    metrics : Iterable[str],
    tune : Optional[dict] = None,
    seed : int = 42,
    **params
):
    ''' Find an approximate best score from clf on X and y.
    '''
    logging.debug(f'Training classifier: {Classifier.__name__}')
    clf = Classifier(**params)
    results = []

    if tune:
        format_params(tune)
        search = GridSearchCV(clf, tune, scoring=metrics, refit=False)
        search.fit(X, y)

        df = pd.DataFrame(search.cv_results_)
        # for each metric choose first row where rank_test_metric is 1
        for metric in metrics:
            best = df[df[f'rank_test_{metric}'] == 1].iloc[0]
            clf.set_params(**best['params'])
            clf.fit(X_test, y_test)
            y_true, y_pred = y_test, clf.predict(X_test)

            metric_results = {'model': Classifier.__name__}
            metric_results.update({'mean_time': best['mean_fit_time'],  'std_time': best['std_fit_time'], 
                'params': best['params']})
            metric_results.update({f'mean_{m}': best[f'mean_test_{m}'] for m in metrics})
            metric_results.update({f'std_{m}': best[f'std_test_{m}'] for m in metrics})
            metric_results.update({f'test_{m}' : get_scorer(m)._score_func(y_true, y_pred) for m in metrics})
            results.append(metric_results)
    else:
        cv_results = cross_validate(clf, X, y, scoring=metrics)
        clf.fit(X, y)
        y_true, y_pred = y_test, clf.predict(X_test)

        tmp = {'model': Classifier.__name__}
        tmp.update(mean_and_std_(cv_results['fit_time'], 'time'))
        for metric in metrics:
            tmp.update(mean_and_std_(cv_results[f'test_{metric}'], metric))

            score = get_scorer(metric)._score_func(y_true, y_pred)
            tmp[f'test_{metric}'] = score

        results.append(tmp)
    
    return results



def get_models_(models : Union[Iterable[Union[str,dict]], str]):
    if models == 'all':
        return list(zip(CLASSIFIER_MAP_.values(), [None]*len(CLASSIFIER_MAP_)))
    elif models == 'all-tuned':
        return [(CLASSIFIER_MAP_[name], TUNING_PARAMETERS_MAP_[name]) for name in CLASSIFIER_MAP_.keys()]
    else:
        all_models = []
        for model in models:
            model_name = model if isinstance(model, str) else model['name']
            params = None if isinstance(model, str) else without(model, 'name')
            all_models.append((CLASSIFIER_MAP_[model_name], params))
        return all_models


def train_classifiers(
    dataset : Dataset, 
    models : Union[Iterable[Union[str,dict]], str], 
    X_columns : Optional[Union[str, Iterable[str]]] = None, 
    y_columns : Optional[Union[str, Iterable[str]]] = None, 
    metrics : Iterable[str] = ['accuracy'],
    **kwargs
):
    ''' Train each model on the dataset and return the best for each model.
    '''
    if X_columns is None and y_columns is None:
        raise ValueError('Must provide at least 1 of \'X_columns\' or \'y_columns\'.')
    
    if X_columns is None:
        X_columns = dataset.all_columns_except(y_columns)
    if y_columns is None:
        y_columns = dataset.all_columns_except(X_columns)

    y_columns = unlistify(y_columns)
    X_train, y_train = dataset.train[X_columns], dataset.train[y_columns]
    X_test, y_test = dataset.test[X_columns], dataset.test[y_columns]

    results = []
    models = get_models_(models)
    for Clf, params in alive_it(models, title='Training'):
        scores = get_classifier_best_(Clf, X_train, y_train, X_test, y_test, metrics, tune=params)
        results.extend(scores)

    return pd.DataFrame(results)
