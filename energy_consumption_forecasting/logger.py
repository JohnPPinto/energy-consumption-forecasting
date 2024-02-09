import logging
import os
from datetime import datetime
from pathlib import Path

from energy_consumption_forecasting.utils import get_env_var

ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))
LOG_PATH = ROOT_DIRPATH / "logs"
FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILENAME = LOG_PATH / FILE_NAME

if not os.path.isdir(LOG_PATH):
    os.makedirs(name=LOG_PATH)


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
