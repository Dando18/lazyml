""" Preprocessing functions.
    author: Daniel Nichols
    date: February, 2022
"""
# std imports
from typing import Iterable


def scale(dataset, columns: Iterable[str], **kwargs):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(dataset.train[columns])
    dataset.train[columns] = scaler.transform(dataset.train[columns])
    if dataset.has_testing_set():
        dataset.test[columns] = scaler.transform(dataset.test[columns])


def minmax_scale(dataset, columns: Iterable[str], **kwargs):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(**kwargs)
    scaler.fit(dataset.train[columns])
    dataset.train[columns] = scaler.transform(dataset.train[columns])
    if dataset.has_testing_set():
        dataset.test[columns] = scaler.transform(dataset.test[columns])


def maxabs_scale(dataset, columns: Iterable[str], **kwargs):
    from sklearn.preprocessing import MaxAbsScaler

    scaler = MaxAbsScaler(**kwargs)
    scaler.fit(dataset.train[columns])
    dataset.train[columns] = scaler.transform(dataset.train[columns])
    if dataset.has_testing_set():
        dataset.test[columns] = scaler.transform(dataset.test[columns])


def normalize(dataset, columns: Iterable[str], **kwargs):
    from sklearn.preprocessing import Normalizer

    scaler = Normalizer(**kwargs)
    dataset.train[columns] = scaler.fit_transform(dataset.train[columns])
    if dataset.has_testing_set():
        dataset.test[columns] = scaler.transform(dataset.test[columns])


def encode(dataset, columns: Iterable[str], **kwargs):
    from sklearn.preprocessing import LabelEncoder

    for column in columns:
        encoder = LabelEncoder()
        encoder.fit(dataset.train[column])

        dataset.test = dataset.test[dataset.test[column].isin(encoder.classes_)]

        dataset.train[column] = encoder.transform(dataset.train[column])
        if dataset.has_testing_set():
            dataset.test[column] = encoder.transform(dataset.test[column])


def one_hot_encode(dataset, columns: Iterable[str], **kwargs):
    dataset.one_hot_encode(columns)


def binarize(dataset, columns: Iterable[str], threshold=0.0):
    from sklearn.preprocessing import Binarizer

    binarizer = Binarizer(threshold=threshold)
    binarizer.fit(dataset.train[columns])

    dataset.train[columns] = binarizer.transform(dataset.train[columns])
    if dataset.has_testing_set():
        dataset.test[columns] = binarizer.transform(dataset.test[columns])


PREPROCESSORS_MAP_ = {
    "scale": scale,
    "minmax_scale": minmax_scale,
    "maxabs_scale": maxabs_scale,
    "normalize": normalize,
    "encode": encode,
    "one_hot_encode": one_hot_encode,
    "binarize": binarize,
}


def preprocess(name, dataset, **kwargs):
    PREPROCESSORS_MAP_[name](dataset, **kwargs)
