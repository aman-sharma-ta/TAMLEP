import argparse
import os
import tarfile
from datetime import datetime

import config
import numpy as np
import pandas as pd
from logging_util import configure_logger
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def fetch_housing_data(
    housing_url=config.housing_url,
    housing_path=config.housing_raw_path,
):
    """This function takes parameters housing_url and housing_path
    Downloads the data from housing_url, extracts the .tgz file and saves
    it to housing_path

    Args:
        housing_url (str, optional):URL from which housing data is downloaded.
        Defaults to config.housing_url.
        housing_path (str, optional): output path to store the downloaded data.
        Defaults to config.housing_raw_path.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=config.housing_raw_path):
    """This function use to load the data and returns csv

    Args:
        housing_path (str, optional): _description_.
        Defaults to config.housing_raw_path.

    Returns:
        csv: dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_train_test_split(housing_data=None):
    """Preprocessing of data and split into train and test dataset
    traget_variable: median_house_value

    Args:
        housing_data (dataframe, optional): csv file to process.
        Defaults to None.
    Returns:
        train_set,test_set
    """
    housing = housing_data.copy()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    housing_final = housing_prepared.join(housing_labels)

    train_set, test_set = train_test_split(
        housing_final, test_size=0.2, random_state=42
    )

    os.makedirs(config.train_housing_path, exist_ok=True)
    os.makedirs(config.test_housing_path, exist_ok=True)
    train_set.to_csv(
        os.path.join(config.train_housing_path, "train.csv"),
        index=False,
    )
    test_set.to_csv(
        os.path.join(config.test_housing_path, "test.csv"),
        index=False,
    )
    return train_set, test_set


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_file_path",
        help="Log file path",
        default=config.log_path,
    )
    parser.add_argument(
        "--log_file_name",
        help="Log file_name",
        default="ingest_data.log",
    )
    # parser.add_argument('data_path',help='data path',
    # default=config.housing_raw_path)
    parser.add_argument(
        "--train_data_path",
        help="test data path",
        default=config.train_housing_path,
    )
    parser.add_argument(
        "--test_data_path",
        help="test data path",
        default=config.test_housing_path,
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

    logger.info("output to console {}".format(args.console.upper()))
    logger.info(
        "fetching housing data from {}".format(config.housing_url)
    )
    fetch_housing_data()
    logger.info("fetching data completed")
    housing = load_housing_data()
    logger.info("train_data_path:{}".format(args.train_data_path))
    logger.info("test_data_path:{}".format(args.test_data_path))
    logger.info("fetched data size {}".format(housing.shape))
    logger.info("starting train-test split with test size 0.2")
    train_set, test_set = data_train_test_split(housing)
    logger.info(
        "completed train-test split with train_size {} and test_size {}".format(
            train_set.shape, test_set.shape
        )
    )
    end = datetime.now()
    logger.info(
        "execution time for ingest_data script {}s".format(
            round((end - start).seconds, 4)
        )
    )
