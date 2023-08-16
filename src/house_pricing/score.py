"""
This script is use to predict the model evaluation on test data.

Input:
    score.py [-h] [--test_data_path TEST_DATA_PATH]
                    [--model_folder MODEL_FOLDER] [--log_file_path LOG_FILE_PATH]
                    [--log_file_name LOG_FILE_NAME] [--console {Y,N,y,n}]
                    [--log_level {DEBUG,INFO,CRITICAL,ERROR,WARNING}]
                    model_name
Output:
    Predicted result with true value stored in output directory
"""
import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from logging_util import configure_logger
except ImportError:
    print("oh no config file")


def predict_result(model_path=None, test_data_path=None, logger=None):
    """This function is use to predict result on the test_data and
    returns predicted_result

    Args:
        test_data_path (str, optional):
        test data set file path (.csv).
        Defaults: None.

        model_path (str):
        trained model file.
        Defaults: None.

        logger (log_obj):
        it will help to log data in the log_file.
        Defaults: None.

    Returns:
        predicted_data:Dataframe
    """
    data = pd.read_csv(test_data_path)
    y_test = data["median_house_value"]
    x_test = data.drop("median_house_value", axis=1)

    logger.info("data extracted {}".format(data.shape))

    if x_test.shape[0] == y_test.shape[0]:
        logger.info("test data check passed")
    else:
        logger.info("test data check failed")

    model = joblib.load(model_path)

    model_name = model_path.split("/")[-1].split(".pkl")[0]

    logger.info("model picked : {}".format(model_name))

    pred_result = model.predict(x_test)
    logger.info("prediction done for x_test")

    if "rfr" in model_name:
        cvres = model.cv_results_
        for mean_score, params in zip(
            cvres["mean_test_score"], cvres["params"]
        ):
            logger.info(
                "model_name:{},rmse_score: {},params: {}".format(
                    model_name, np.sqrt(-mean_score), params
                )
            )
    else:
        result_mse = mean_squared_error(y_test, pred_result)
        result_rmse = np.sqrt(result_mse)
        result_mae = mean_absolute_error(y_test, pred_result)
        logger.info(
            "model_name: {},rmse_score: {},mae_score: {}".format(
                model_name, result_rmse, result_mae
            )
        )
    data[model_name + "_predictions"] = pred_result
    return data


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", help="model name required")
    parser.add_argument(
        "--test_data_path",
        help="test data folder",
        default=os.path.join(config.test_housing_path, "test.csv"),
    )
    parser.add_argument(
        "--model_folder",
        help="path to model folder",
        default=config.artifacts_path,
    )
    parser.add_argument(
        "--log_file_path",
        help="Log file path",
        default=config.log_path,
    )
    parser.add_argument(
        "--log_file_name", help="Log file_name", default="score.log"
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
    logger.info("Scoring Starts")
    logger.info(
        "log_file_location:{}/{}".format(
            args.log_file_path, args.log_file_name
        )
    )
    logger.info("test_file_loc:{}".format(args.test_data_path))
    logger.info("model folder location {}".format(args.model_folder))
    logger.info("model_selection:{}".format(args.model_name))
    logger.info("output_path:{}".format(config.output_path))

    predicted_data = predict_result(
        os.path.join(args.model_folder, args.model_name),
        args.test_data_path,
        logger,
    )

    os.makedirs(config.output_path, exist_ok=True)

    predicted_data.to_csv(
        os.path.join(
            config.output_path,
            args.model_name.split(".pkl")[0] + "_predictions.csv",
        ),
        index=False,
    )
    logger.info(
        "prediction result saved :{}".format(
            os.path.join(
                config.output_path,
                args.model_name.split(".pkl")[0] + "_predictions.csv",
            )
        )
    )
    logger.info("Scoring Ends")
    end = datetime.now()
    logger.info(
        "execution time for ingest_data script {}s".format(
            round((end - start).seconds, 4)
        )
    )
