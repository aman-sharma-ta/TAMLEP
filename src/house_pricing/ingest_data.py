import os
import tarfile

import config
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def fetch_housing_data(
    housing_url=config.housing_url, housing_path=config.housing_raw_path
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
        os.path.join(config.train_housing_path, "train.csv"), index=False
    )
    test_set.to_csv(
        os.path.join(config.test_housing_path, "test.csv"), index=False
    )


if __name__ == "__main__":
    # fetch_housing_data()
    housing = load_housing_data()
    data_train_test_split(housing)
