"""
"""
# std imports
from argparse import ArgumentParser
import json
import logging

# tpl imports
from alive_progress import alive_it
from mpi4py import MPI

# local imports
from lazyml.util import without, parse_columns
from lazyml.dataset import get_dataset
from lazyml.preprocess import preprocess
from lazyml.train_classifiers import train_classifiers
from lazyml.train_regressors import train_regressors
from lazyml.train_clustering import train_clustering

# turn off sklearn warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args():
    parser = ArgumentParser(description="Brute force ML tool")
    parser.add_argument("config", type=str, help="config json")
    parser.add_argument("-o", "--output", type=str, help="output path")
    parser.add_argument('-p', '--parallel', action='store_true', help='compute in parallel with MPI')
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

    if args.parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    if not args.parallel or rank == 0:
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
    logging.info("Training{}...".format(f" on rank {rank}" if args.parallel else ""))
    training_config = config["train"]
    if "X" in training_config and "y" in training_config:   # x and y
        X_columns = parse_columns(training_config['X'], dataset)
        y_columns = parse_columns(training_config['y'], dataset)
    elif "X" in training_config:    # just x
        X_columns = parse_columns(training_config['X'], dataset)
        y_columns = dataset.all_columns_except(X_columns)
    elif "y" in training_config:    # just y
        y_columns = parse_columns(training_config['y'], dataset)
        X_columns = dataset.all_columns_except(y_columns)
    else:   # no x or y
        raise ValueError("Must provide at least 1 of 'X' or 'y' for training.")

    if training_config["task"] == "classification":
        results = train_classifiers(
            dataset,
            X_columns=X_columns,
            y_columns=y_columns,
            seed=args.seed,
            dim_reduce_config=dim_reduce_config,
            **without(training_config, "task", "X", "y"),
        )
    elif training_config["task"] == "regression":
        results = train_regressors(
            dataset,
            X_columns=X_columns,
            y_columns=y_columns,
            seed=args.seed,
            dim_reduce_config=dim_reduce_config,
            **without(training_config, "task", "X", "y"),
        )
    elif training_config["task"] == "clustering":
        results = train_clustering(
            dataset,
            X_columns=X_columns,
            y_columns=y_columns,
            seed=args.seed,
            dim_reduce_config=dim_reduce_config,
            **without(training_config, "task", "X", "y"),
        )
    logging.info("Done training.")

    # log best
    result_columns = results.columns[results.columns.str.startswith("test_")]
    for metric in result_columns:
        metric_name = metric[len("test_") :]
        best_row = results.loc[results[metric].idxmax()]
        n_others = (
            len(results[results[metric] == best_row[metric]]["model"].to_list()) - 1
        )
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
