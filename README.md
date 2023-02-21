# LazyML

![Status Badge](https://github.com/Dando18/lazyml/actions/workflows/lazyml-test/badge.svg)

A transparent autoML-like tool.
It is meant to make testing a lot of different models on a dataset easier.
It is not meant to be a black-box prediction tool.

- [LazyML](#lazyml)
- [Usage](#usage)
- [Config File Format](#config-file-format)
  - [data](#data)
  - [preprocess](#preprocess)
  - [dimensionality\_reduction](#dimensionality_reduction)
  - [train](#train)
    - [Classification Models](#classification-models)
    - [Regression Models](#regression-models)

# Usage
The script takes a configuration _json_ file to define data, preprocessing,
and training parameters.
It is run as `python3 lazyml.py <config.json>`.
See [examples dir](examples/) for example config files.
Here's a simple classification config for the UCI iris dataset:

```json
{
    "data": {
        "train": "./datasets/iris.csv",
        "test_split": "0.2",
        "drop": ["Id"]
    },
    "preprocess": [
        {"name": "scale", "columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]}
    ],
    "train": {
        "task": "classification",
        "X_columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "y_columns": ["Species"],
        "metrics": ["accuracy"],
        "models": "all-tuned"
    }
}
```

# Config File Format

## data
`"data": {...}`

Defines what data to load and some initial preprocessing steps. Options are:

- `"train":` training dataset; path/url to CSV file or list of paths and urls
- `"test":` or `"test_split":` testing dataset; give paths/urls to CSV files or define _test\_split_ to use a portion of the training dataset for testing
- `"drop":` a list of column names to drop
- `"dropna":` a list of column names to drop _Na_ values from

## preprocess
`"preprocess": [...]`

_Optional step_.
Defines a list of preprocessing steps. 
Each preprocessing step is formatted as `{"name": ..., "columns": [...], ...}`.
Possible preprocessing names are:

- `"scale"`: scale each column by _(x\_i - mean) / std_. See [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- `"minmax_scale"`: scale each column by _(x\_i - min(x)) / (max(x) - min(x))_. See [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
- `"maxabs_scale"`: scale each column by _(x\_i / max(abs(x)))_. See [sklearn.preprocessing.MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).
- `"normalize"`: normalize each sample. See [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).
- `"encode"`: encode column with integer labels _0...n-1_. See [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
- `"one_hot_encode"`: one-hot-encode columns. See [pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
- `"binarize"`: Binarize values by some threshold. See [sklearn.preprocessing.Binarize](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarize.html).

The columns to apply the preprocessing step are given by `"columns"` or `"all-columns-except"`. The latter means to take the converse subset of column names and apply to those.

Any additional arguments are passed to the underlying preprocessing method. For instance, `norm="l2"` or `norm="l1"` can be passed to `"normalize"` with an additional entry `"norm": "l1"`.

## dimensionality_reduction
`"dimensionality_reduction": {...}`

_Optional step_. Defines how to reduce feature dimensions if at all. Options are:

- `{"name": "PCA", "n_components": ..., ...}` Principal Component Analysis to _n\_components_. See [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- `{"name": "LDA", "n_components": ..., ...}` Linear Discriminant Analysis to _n\_components_. Must be classification training task and the number of components must be strictly less than the number of classes. See [sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

## train
`"train": {...}`

Defines a list of training tasks to try on the data. 
The type of training task is given by `"task": ...` and can be one of

- `"classification"`  Train classifier(s) on the dataset.
- `"regression"` Train regressors on the dataset.

The inputs to the model, _X_, and output targets, _y_, are given by `"X_columns": [...]` and `"y_columns": [...]`. If one is omitted, then it is assumed to be the converse of the other.

Training objectives are given by the `"metrics": [...]` keyword.
Each of these are recorded during training and the best model for
each metric is returned. See [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values) page for possible metrics for each training task.

Finally, `"models": [...]` is used to define the list of machine learning models to train. `"models": "all"` will train all the available models for that task and `"models": "all-tuned"` will do the same with hyperparameter tuning.
When defining individual models you can either use the name of the model or a dict with parameters. For instance, the following will train the _Dummy_, _DecisionForest_, _SVM_, and _AdaBoost_ models. _SVM_ has its _C_ parameter passed as 0.1. _AdaBoost_ defines a range of values for _n\_estimators_ and will, thus, do hyperparameter tuning.

```json
"models": [
    "Dummy",
    "DecisionForest",
    {"name": "SVM", "C": 0.1},
    {"name": "AdaBoost", "n_estimators": [10, 100, 200]}
]
```

The available models for each training task are listed below.

### Classification Models

- `"Dummy"` See [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- `"Ridge"` See [sklearn.linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- `"DecisionTree"` See [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- `"SGD"` See [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- `"AdaBoost"` See [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- `"GradientBoosting"` See [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- `"RandomForest"` See [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- `"GaussianProcess"` See [sklearn.gaussian_process.GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)
- `"Perceptron"` See [sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- `"GaussianNB"` See [sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- `"KNeighbors"` See [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- `"MLP"` See [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- `"SVM"` See [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- `"LinearSVM"` See [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- `"XGB"` See [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)

### Regression Models

- `"Dummy"` See [sklearn.dummy.DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- `"Linear"` See [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- `"Ridge"` See [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- `"Lasso"` See [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- `"SGD"` See [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- `"PassiveAggressive"` See [sklearn.linear_model.PassiveAggressiveRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)
- `"Poisson"` See [sklearn.linear_model.PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- `"KNeighbors"` See [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- `"AdaBoost"` See [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
- `"GradientBoosting"` See [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- `"RandomForest"` See [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- `"DecisionTree"` See [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- `"MLP"` See [sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- `"SVM"` See [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- `"LinearSVM"` See [sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
- `"XGB"` See [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)