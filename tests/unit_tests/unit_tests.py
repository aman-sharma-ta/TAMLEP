import os
import tarfile
import pandas as pd
from pandas.testing import assert_frame_equal
from house_pricing import ingest_data, config, train_data

os.makedirs("tests/tmp_dir/", exist_ok=True)

def test_fetch_housing_data(tmp_path="tests/tmp_dir"):
    """
    Function to test fetch_housing_data
    """
    # Set up test variables
    url = config.housing_url
    data_dir = tmp_path + "/data"

    # Call the function being tested
    ingest_data.fetch_housing_data(url, data_dir)

    # Assert that the data was downloaded and extracted
    assert os.path.isdir(data_dir), "Folder not created to download data!!"
    assert os.path.isfile(
        data_dir + "/housing.tgz"
    ), "Data not downloaded from the given URL!!"
    assert tarfile.is_tarfile(
        data_dir + "/housing.tgz"
    ), "downloaded data format does not match tar file!!"
    assert os.path.isfile(
        data_dir + "/housing.csv"
    ), "Downloaded data was not extracted properly!!"

