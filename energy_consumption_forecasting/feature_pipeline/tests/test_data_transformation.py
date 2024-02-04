import numpy as np
import pandas as pd
import pytest

from energy_consumption_forecasting.feature_pipeline.data_transformation import (
    casting_features,
    clean_dataframe,
    feature_engineering,
    rename_features,
)


# Creating a dummy dataframe for testing the functions
@pytest.fixture
def get_dataframe():
    data = {
        "HourUTC": [
            "2023-06-30T22:00:00",
            "2023-07-01T23:00:00",
            "2023-07-02T00:00:00",
            "2023-06-30T22:00:00",
            "2023-07-04T02:00:00",
        ],
        "HourDK": [
            "2023-07-01T00:00:00",
            "2023-07-02T01:00:00",
            "2023-07-03T02:00:00",
            "2023-07-01T00:00:00",
            "2023-07-05T04:00:00",
        ],
        "MunicipalityNo": [
            250,
            773,
            766,
            250,
            np.nan,
        ],
        "Branche": [
            "Offentligt",
            "Erhverv",
            "Privat",
            "Offentligt",
            "Erhverv",
        ],
        "ConsumptionkWh": [
            1361.189,
            7812.050,
            6760.285,
            1361.189,
            72003.364,
        ],
    }

    return pd.DataFrame(data)


def test_clean_dataframe(get_dataframe):
    """
    In this test the dataframe is cleaned and returned as pandas dataframe.
    """

    result_df = clean_dataframe(dataframe=get_dataframe)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (3, 5)

    # Dropping a column
    result_df = clean_dataframe(dataframe=get_dataframe, drop_columns=["HourUTC"])
    assert result_df.shape == (3, 4)

    with pytest.raises(Exception) as exe_info:
        result_df = clean_dataframe(dataframe=get_dataframe, drop_columns=[])
        assert "Empty list argument is not allowed in drop_columns." in str(
            exe_info.value
        )

        result_df = clean_dataframe(dataframe=get_dataframe, drop_columns=[123])
        assert (
            'Column "123" does not exist in the dataFrame, '
            "kindly recheck the column names to drop it." in str(exe_info.value)
        )


def test_rename_features(get_dataframe):
    """
    In this test the dataframe columns are renamed and returned as pandas dataframe.
    """

    rename_dict = {
        "HourDK": "datetime_dk",
        "MunicipalityNo": "municipality_num",
    }
    result_df = rename_features(
        dataframe=get_dataframe, rename_columns_dict=rename_dict
    )
    assert isinstance(result_df, pd.DataFrame)
    assert (
        "HourDK" not in result_df.columns and "MunicipalityNo" not in result_df.columns
    )
    assert (
        "datetime_dk" in result_df.columns and "municipality_num" in result_df.columns
    )

    with pytest.raises(Exception) as exe_info:
        result_df = rename_features(dataframe=get_dataframe, rename_columns_dict={})
        assert "Empty Dict argument is not allowed in rename_columns_dict." in str(
            exe_info.value
        )

        result_df = rename_features(
            dataframe=get_dataframe, rename_columns_dict={"Hour": "datetime"}
        )
        assert (
            'Column "Hour" does not exist in the dataFrame, '
            "kindly recheck the column names to rename it." in str(exe_info.value)
        )


def test_casting_features(get_dataframe):
    """
    In this test the data type of dataframe features are transformed
    and returned as pandas dataframe.
    """

    df = clean_dataframe(dataframe=get_dataframe)
    rename_dict = {
        "HourDK": "datetime_dk",
        "MunicipalityNo": "municipality_num",
        "Branche": "branch",
        "ConsumptionkWh": "consumption_kwh",
    }
    df = rename_features(dataframe=df, rename_columns_dict=rename_dict)
    result_df = casting_features(dataframe=df)

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(result_df.datetime_dk.dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(result_df.municipality_num.dtype, type(np.dtype("int32")))
    assert isinstance(result_df.branch.dtype, pd.StringDtype)
    assert isinstance(result_df.consumption_kwh.dtype, type(np.dtype("float64")))


def test_feature_engineering(get_dataframe):
    """
    In this test the dataframe perform feature engineering and returns the dataframe
    """
    df = rename_features(
        dataframe=get_dataframe, rename_columns_dict={"Branche": "branch"}
    )
    result_df = feature_engineering(dataframe=df)

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(result_df.branch.dtype, type(np.dtype("int8")))
    assert len(result_df.branch.unique()) == 3
    assert all(result_df.branch.unique() != ["Offentligt", "Erhverv", "Privat"])
    assert all(result_df.branch.unique() == [1, 2, 3])
