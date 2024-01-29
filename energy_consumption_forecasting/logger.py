import logging
import os
from datetime import datetime

FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")

if not os.path.isdir(LOG_PATH):
    os.makedirs(name=LOG_PATH)

LOG_FILENAME = os.path.join(LOG_PATH, FILE_NAME)


def get_logger(name: str = "") -> logging.Logger:
    """
    This function provides a custom template for logging
    the details in the logs directory.

    Parameters
    ----------
    name: str, default = ""
        Name of the logger, eg. root or __main__

    Returns
    -------
    logger: logging.Logger
        returns a initiated logger object.
    """

    logging.basicConfig(
        filename=LOG_FILENAME,
        format=("[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s"),
        level=logging.INFO,
    )

    logger = logging.getLogger(name=name)

    return logger
