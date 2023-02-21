""" Preprocessing functions.
    author: Daniel Nichols
    date: February, 2022
"""


def normalize(dataset, columns, **kwargs):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(dataset.train[columns])

    dataset.train[columns] = scaler.transform(dataset.train[columns])
    dataset.test[columns] = scaler.transform(dataset.test[columns])


def encode(dataset, columns, **kwargs):
    from sklearn.preprocessing import LabelEncoder

    for column in columns:
        encoder = LabelEncoder()
        encoder.fit(dataset.train[column])

        dataset.test = dataset.test[dataset.test[column].isin(encoder.classes_)]

        dataset.train[column] = encoder.transform(dataset.train[column])
        dataset.test[column] = encoder.transform(dataset.test[column])


def one_hot_encode(dataset, columns, **kwargs):
    dataset.one_hot_encode(columns)


def binarize(dataset, columns, threshold=0.0):
    from sklearn.preprocessing import Binarizer

    binarizer = Binarizer(threshold=threshold)
    binarizer.fit(dataset.train[columns])

    dataset.train[columns] = binarizer.transform(dataset.train[columns])
    dataset.test[columns] = binarizer.transform(dataset.test[columns])


PREPROCESSORS_MAP_ = {
    "normalize": normalize,
    "encode": encode,
    "one_hot_encode": one_hot_encode,
    "binarize": binarize,
}


def preprocess(name, dataset, **kwargs):
    PREPROCESSORS_MAP_[name](dataset, **kwargs)
