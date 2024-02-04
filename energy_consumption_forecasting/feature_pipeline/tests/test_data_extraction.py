import datetime
from pathlib import Path

import pandas as pd
import pytest

from energy_consumption_forecasting.feature_pipeline.data_extraction import (
    extract_dataset_from_api,
    get_extraction_datetime,
)


@pytest.fixture
def date_range():
    start_date = datetime.datetime(2021, 1, 1, 12, 45)
    end_date = datetime.datetime(2021, 1, 10, 5, 10)
    return start_date, end_date


def test_valid_date_range_and_format(date_range):
    """
    In this test, both the start and end date are accurate.
    """
    result = get_extraction_datetime(date_range[0], date_range[1])

    assert isinstance(result, tuple)
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    assert len(result) == 2
    assert result[0] < result[1]

    expected_format = "%Y-%m-%dT%H:%M"
    assert result[0] == date_range[0].strftime(expected_format)
    assert result[1] == (date_range[1] + datetime.timedelta(days=1)).strftime(
        expected_format
    )


def test_invalid_date_range(date_range):
    """
    In this test, the end date is smaller than the start date.
    """
    with pytest.raises(Exception) as exe_info:
        get_extraction_datetime(date_range[1], date_range[0])

    assert "End date needs to be greater than the start date" in str(exe_info.value)

    with pytest.raises(Exception) as exe_info:
        extract_dataset_from_api(date_range[1], date_range[0])

    assert "End date needs to be greater than the start date" in str(exe_info.value)


@pytest.mark.parametrize("save_data", [(False), (True)])
def test_extract_dataset_from_api(date_range, save_data):
    """
    Testing the function "extract_dataset_from_api()" whether it returns data from an
    API and save it in the directory.
    """

    result = extract_dataset_from_api(
        start_date_time=date_range[0],
        end_date_time=date_range[1],
        save_dataset_metadata=save_data,
    )
    if save_data:
        assert len(result) == 4
        assert isinstance(result[2], Path)
        assert isinstance(result[3], Path)
        assert result[2].is_file() == True
        assert result[3].is_file() == True

        # Deleting the generated files
        result[2].unlink()
        result[3].unlink()

    else:
        assert len(result) == 2
        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)
