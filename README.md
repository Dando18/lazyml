# LazyML

A transparent autoML-like tool.
Meant to make testing a lot of different models on a dataset easier.
Not meant to be a black-box prediction tool.

## Usage
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
        {"name": "normalize", "columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]}
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
