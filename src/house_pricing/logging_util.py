import logging
import os


def configure_logger(log_file=None, log_level="DEBUG"):
    """
    Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level`
    will be passed to the function

    Parameters
    ----------
    log_file: str
        Path to the log file for logs to be stored
    log_level: str
        One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
        default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    log_file_path = "/".join(log_file.split("/")[:-1])
    os.makedirs(log_file_path, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s\
 - %(funcName)s:%(lineno)d\
 - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    logger = logging.getLogger()
    return logger
