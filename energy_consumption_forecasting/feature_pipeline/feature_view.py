import datetime
from pathlib import Path
from typing import Dict, Optional

import hopsworks
import hsfs

from energy_consumption_forecasting.exception import log_exception
from energy_consumption_forecasting.logger import get_logger

logger = get_logger(name=Path(__file__).name)


def create_feature_view(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    feature_group_version: int,
) -> Dict:
    """ """

    pass
