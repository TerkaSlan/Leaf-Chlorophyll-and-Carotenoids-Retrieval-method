import argparse
import logging
import os
from datetime import datetime

from tqdm import tqdm

from Experiment import get_features, models
from ExperimentFull import ExperimentFull
from ExperimentSubset import ExperimentSubset
from ExperimentValid import ExperimentValid
from helpers import load_data_LUT, load_valid_dataset


def load_data(data_dir, data_valid_dir, vegetation_parameter, test_run):
    data_train, data_test, params_train, params_test, param_min, param_max = (
        load_valid_dataset(
            data_valid_dir,
            vegetation_parameter,
        )
    )

    # load all the files if we're not using a test run, otherwise load 10
    n_files = None if not test_run else 10
    X_, y_ = load_data_LUT(data_dir, vegetation_parameter, n_files=n_files)
    X_ *= 10_000
    return (
        X_,
        y_,
        data_train,
        data_test,
        params_train,
        params_test,
        param_min,
        param_max,
    )


def run_experiment(
    args,
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    param_min,
    param_max,
    model,
    name,
):
    if args.mode == "subset":
        e = ExperimentSubset(
            args,
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            param_min,
            param_max,
            model,
            name,
        )
    elif args.mode == "full":
        e = ExperimentFull(
            args,
            X_train,
            X_test,
            y_train,
            y_test,
            param_min,
            param_max,
            model,
            name,
        )
    elif args.mode == "valid":
        e = ExperimentValid(
            args,
            X_train,
            X_test,
            y_train,
            y_test,
            param_min,
            param_max,
            model,
            name,
        )
    else:
        raise ValueError("Invalid mode")

    if not e.is_finished:
        e.run()
        e.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # python run.py --param=Cab --top_n=10 --save_dir=results_LUT_sampling_new --mode=subset
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state parameter"
    )
    parser.add_argument("--param", type=str, default="Car", help="Vegetation parameter")
    parser.add_argument(
        "--top_n", type=int, default=10, help="How many items to use in an ensemble"
    )
    parser.add_argument("--save_dir", type=str, help="dir to save to")
    parser.add_argument("--data_dir", type=str, default="data/LUT", help="data dir")
    parser.add_argument(
        "--data_valid_dir", type=str, default="data/valid", help="data valid dir"
    )
    parser.add_argument(
        "--mode", type=str, choices=["full", "subset", "valid"], help="data valid dir"
    )
    parser.add_argument("--test_run", type=bool, default=False)

    args = parser.parse_args()

    random_state = args.random_state
    vegetation_parameter = args.param
    top_n = args.top_n
    k = -1
    savedir = args.save_dir + "_" + vegetation_parameter
    os.environ["SAVEDIR"] = savedir
    if not os.path.exists(os.environ["SAVEDIR"]):
        os.makedirs(os.environ["SAVEDIR"])
        os.makedirs(os.environ["SAVEDIR"] + "/models")
        os.makedirs(os.environ["SAVEDIR"] + "/results")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    log_filename = f"exp_log_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args.k = None
    args.preprocessing_method = None

    X_valid = None
    y_valid = None
    if args.mode == "subset":
        X_train, y_train, X_valid, X_test, y_valid, y_test, param_min, param_max = (
            # (data_dir, data_valid_dir, vegetation_parameter, test_run):
            load_data(args.data_dir, args.data_valid_dir, args.param, args.test_run)
        )
        logging.info("Loaded data, shape: {}, {}".format(X_train.shape, y_train.shape))
    elif args.mode == "full":
        X_train, y_train, _, X_test, _, y_test, param_min, param_max = load_data(
            args.data_dir, args.data_valid_dir, args.param, args.test_run
        )
        logging.info("Loaded data, shape: {}, {}".format(X_train.shape, y_train.shape))
    else:
        X_train, X_test, y_train, y_test, param_min, param_max = load_valid_dataset(
            args.data_valid_dir, vegetation_parameter
        )
        logging.info("Loaded data, shape: {}, {}".format(X_train.shape, y_train.shape))

    for n_features in range(1, X_test.shape[1] + 1):
        features = get_features(args.param, n_features)
        args.features = features
        logging.info(f"Using features: {features}")

        X_train_ = X_train[features].copy()
        X_test_ = X_test[features].copy()
        X_valid_ = X_valid[features].copy() if X_valid is not None else None

        for preprocessing_method in [None, "StandardScaler", "PCA", "SVD"]:
            args.preprocessing_method = preprocessing_method
            if preprocessing_method == "PCA" or preprocessing_method == "SVD":
                for k in [4, 5, 6]:
                    if k > n_features:
                        continue
                    for name, model in tqdm(models):
                        args.k = k
                        run_experiment(
                            args,
                            X_train_,
                            X_valid_,
                            X_test_,
                            y_train,
                            y_valid,
                            y_test,
                            param_min,
                            param_max,
                            model,
                            name,
                        )
            else:
                for name, model in tqdm(models):
                    run_experiment(
                        args,
                        X_train_,
                        X_valid_,
                        X_test_,
                        y_train,
                        y_valid,
                        y_test,
                        param_min,
                        param_max,
                        model,
                        name,
                    )
