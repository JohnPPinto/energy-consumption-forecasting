import logging
import os
from datetime import datetime

FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")

if not os.path.isdir(LOG_PATH):
    os.makedirs(name=LOG_PATH)

LOG_FILENAME = os.path.join(LOG_PATH, FILE_NAME)

logging.basicConfig(
    filename=LOG_FILENAME,
    format=("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"),
    level=logging.INFO,
)
