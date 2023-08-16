"""
This Script is utilized to train models on the
training data available.

Input:
    usage: train_data.py [-h] [--train_data_path TRAIN_DATA_PATH]
                     [--model_name {lr,dtr,rfr_rs,rfr_gs}]
                     [--output_folder OUTPUT_FOLDER]
                     [--log_file_path LOG_FILE_PATH]
                     [--log_file_name LOG_FILE_NAME] [--console {Y,N,y,n}]
                     [--log_level {DEBUG,INFO,CRITICAL,ERROR,WARNING}]
Ouput:
    trained_model(.pkl) store in artifacts/
"""

import argparse
import os
import sys
from datetime import datetime
from random import randint

import joblib

# import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from logging_util import configure_logger
except ImportError:
    print("oh no config file")


def train_model(
    train_data_set_path=None, model_name=None, logger=None
):
    """This function is use to train model on the data and
    returns trained_model

    Args:
        train_data_set_path (str, optional):
        train data set file path (.csv).
        Defaults to None.

        model_name (str):
        name that you need to pass
        [lr,dtr, rfr_rs,rfr_gs]
        Defaults to None.

        logger (log_obj):
        it will help to log data in the log_file
        Defaults None.

    Returns:
        trained_model
    """

    train_data = pd.read_csv(train_data_set_path)
    y_label = train_data["median_house_value"]
    x_label = train_data.drop("median_house_value", axis=1)

    logger.info("train data extracted {}".format(train_data.shape))
    logger.info(
        "x_data {}, Y_data {}".format(x_label.shape, y_label.shape)
    )
    # test1
    if x_label.shape[0] == y_label.shape[0]:
        logger.info("Passed data check")
    else:
        logger.info("failed data check")

    logger.info("model_selected : {}".format(model_name))

    if model_name == "lr":
        lr = LinearRegression()
        logger.info("Linear model learning starts")
        lr.fit(x_label, y_label)
        logger.info("linear model learning end")
        return lr
    elif model_name == "dtr":
        dtr = DecisionTreeRegressor(random_state=42)
        logger.info("decision tree model train initiated")
        dtr.fit(x_label, y_label)
        logger.info("decision tree model train end")
        return dtr
    elif model_name == "rfr_rs":
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        logger.info("rndm forst rndsrch model train start")
        rnd_search.fit(x_label, y_label)
        logger.info("rndm forst rndsrch model train end")
        return rnd_search
    elif model_name == "rfr_gs":
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {
                "n_estimators": [3, 10, 30],
                "max_features": [2, 4, 6, 8],
            },
            # then try 6 (2×3) combinations with bootstrap set as False
            {
                "bootstrap": [False],
                "n_estimators": [3, 10],
                "max_features": [2, 3, 4],
            },
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        logger.info("rndm forst grdsrch model train start")
        grid_search.fit(x_label, y_label)
        logger.info("rndm forst grdsrch model train end")
        return grid_search
    else:
        return None


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_path",
        help="train data path ",
        default=os.path.join(config.train_housing_path, "train.csv"),
    )

    parser.add_argument(
        "--model_name",
        help="(lr),(dtr),(rfr_rs),(rfr_gs)",
        default="lr",
        choices=["lr", "dtr", "rfr_rs", "rfr_gs"],
    )

    parser.add_argument(
        "--output_folder",
        help="path to output model folder",
        default=config.artifacts_path,
    )

    parser.add_argument(
        "--log_file_path",
        help="Log file path",
        default=config.log_path,
    )
    parser.add_argument(
        "--log_file_name",
        help="Log file_name",
        default="train_data.log",
    )

    parser.add_argument(
        "--console",
        help="whether to output logs to console (Y/N)?",
        default="Y",
        choices=["Y", "N", "y", "n"],
    )
    parser.add_argument(
        "--log_level",
        help="level of logs that required",
        default="DEBUG",
        choices=["DEBUG", "INFO", "CRITICAL", "ERROR", "WARNING"],
    )

    args = parser.parse_args()

    logger = configure_logger(
        os.path.join(args.log_file_path, args.log_file_name),
        args.console.upper(),
        args.log_level,
    )

    logger.info("console output : {}".format(args.console))
    logger.info("Data Training Starts")
    logger.info("train_file_loc:{}".format(args.train_data_path))
    logger.info("model_selection:{}".format(args.model_name))

    model = train_model(args.train_data_path, args.model_name, logger)

    logger.info("data training end")

    s_model_name = (
        args.model_name + "_" + str(datetime.now().date()) + ".pkl"
    )

    joblib.dump(model, os.path.join(args.output_folder, s_model_name))
    logger.info(
        "model_output_loc:{}".format(
            os.path.join(args.output_folder, s_model_name)
        )
    )
    logger.info("model saved completed")
    end = datetime.now()
    logger.info(
        "execution time for ingest_data script {}s".format(
            round((end - start).seconds, 4)
        )
    )
