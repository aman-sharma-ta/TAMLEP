import os

download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housing_url = download_root + "datasets/housing/housing.tgz"
housing_raw_path = os.path.join(os.getcwd(), "data", "raw")
housing_processed_path = os.path.join(os.getcwd(), "data", "processed")
train_housing_path = os.path.join(housing_processed_path, "train")
test_housing_path = os.path.join(housing_processed_path, "test")
log_path = os.path.join(os.getcwd(), "logs")
artifacts_path = os.path.join(os.getcwd(), "artifacts")