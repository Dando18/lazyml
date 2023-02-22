"""
"""
# std imports
from argparse import ArgumentParser
import json
import logging

# tpl imports
from alive_progress import alive_it

# local imports
from lazyml.util import without, parse_columns
from lazyml.dataset import get_dataset
from lazyml.preprocess import preprocess
from lazyml.train_classifiers import train_classifiers
from lazyml.train_regressors import train_regressors

# turn off sklearn warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_args():
    parser = ArgumentParser(description="Brute force ML tool")
    parser.add_argument("config", type=str, help="config json")
    parser.add_argument("-o", "--output", type=str, help="output path")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for numpy/pandas/sklearn"
    )
    parser.add_argument(
        "--log",
        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        type=str.upper,
        help="logging level",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(args.log))
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] -- %(message)s", level=numeric_level
    )

    with open(args.config, "r") as fp:
        config = json.load(fp)

    # read in dataset
    logging.info("Loading dataset...")
    dataset = get_dataset(seed=args.seed, **config["data"])
    logging.info("Dataset loaded.")

    # preprocess dataset
    logging.info("Preprocessing...")
    for preprocess_step in alive_it(config["preprocess"], title="Preprocessing"):
        parse_columns(preprocess_step, dataset)
        preprocess(preprocess_step["name"], dataset, **without(preprocess_step, "name"))
    logging.debug("Post-preprocessing columns: " + ", ".join(dataset.train.columns))
    logging.info("Done preprocessing.")

    # dim reduce parameter
    dim_reduce_config = (
        config["dimensionality_reduction"]
        if "dimensionality_reduction" in config
        else None
    )

    # train
    logging.info("Training...")
    training_config = config["train"]
    if training_config["task"] == "classification":
        results = train_classifiers(
            dataset,
            seed=args.seed,
            dim_reduce_config=dim_reduce_config,
            **without(training_config, "task"),
        )
    elif training_config["task"] == "regression":
        results = train_regressors(
            dataset,
            seed=args.seed,
            dim_reduce_config=dim_reduce_config,
            **without(training_config, "task"),
        )
    logging.info("Done training.")

    # log best
    result_columns = results.columns[results.columns.str.startswith("test_")]
    for metric in result_columns:
        metric_name = metric[len("test_"):]
        best_row = results.loc[results[metric].idxmax()]
        n_others = len(results[results[metric] == best_row[metric]]["model"].to_list())
        logging.info(
            f"{best_row['model']} has the highest {metric_name} at {best_row[metric]} ({n_others} others)"
        )

    # save
    if args.output:
        results.to_csv(args.output)
    else:
        print(results)


if __name__ == "__main__":
    main()
