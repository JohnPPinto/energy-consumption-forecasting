from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import validate_call

from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.logger import get_logger

logger = get_logger(name=Path(__file__).name)


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def clean_dataframe(
    dataframe: pd.DataFrame,
    check_columns_duplicates: List[Any],
    drop_columns: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    This function cleans the provided dataframe by removing missing and duplicate
    data.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas dataframe as a input for transformation.

    check_columns_duplicate: List[Any]
        A list containing column names for checking duplication and dropping the rows.

    drop_columns: List[Any] or None, default=None
        A list containing column names for dropping it from the DataFrame.

    Returns
    -------
    pd.DataFrame
        Returns a cleaned pandas dataframe.
    """
    dataset_df = dataframe.copy()

    # Cleaning the NaN values
    dataset_df.dropna(inplace=True)

    # Dropping unwanted columns
    if drop_columns is not None:
        if len(drop_columns) > 0:
            for col in drop_columns:
                if not col in dataset_df.columns:
                    raise Exception(
                        f'Column "{col}" does not exist in the dataFrame, '
                        "kindly recheck the column names to drop it."
                    )
            dataset_df.drop(columns=drop_columns, inplace=True)
        else:
            raise Exception("Empty list argument is not allowed in drop_columns.")

    # Dropping duplicate values in the dataframe
    dataset_df.drop_duplicates(
        subset=check_columns_duplicates,
        ignore_index=True,
        inplace=True,
    )

    return dataset_df


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def rename_features(
    dataframe: pd.DataFrame,
    rename_columns_dict: Dict[Any, Any],
) -> pd.DataFrame:
    """
    This function rename features of the dataframe
    by using the provided new names in dict format.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas dataframe as a input for transformation.

    rename_columns_dict: Dict[Any, Any]
        A dict containing existing column names as keys and new column names as values.

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe with new feature names.
    """
    dataset_df = dataframe.copy()

    if bool(rename_columns_dict):
        for key in rename_columns_dict.keys():
            if not key in dataset_df.columns:
                raise Exception(
                    f'Column "{key}" does not exist in the dataFrame, '
                    "kindly recheck the column names to rename it."
                )
        dataset_df.rename(columns=rename_columns_dict, inplace=True)
    else:
        raise Exception("Empty Dict argument is not allowed in rename_columns_dict.")

    return dataset_df


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def casting_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function transform features data type of the dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas dataframe as a input for transformation.

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe with transformed feature data types.
    """
    dataset_df = dataframe.copy()

    dataset_df["datetime_dk"] = pd.to_datetime(
        dataset_df["datetime_dk"],
        format="%Y-%m-%dT%H:%M:%S",
    )
    dataset_df["branch"] = dataset_df["branch"].astype("string")
    dataset_df["municipality_num"] = dataset_df["municipality_num"].astype("int32")
    dataset_df["consumption_kwh"] = dataset_df["consumption_kwh"].astype("float64")

    return dataset_df


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs feature engineering on the dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas dataframe as a input for transformation.

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe after performing feature engineering.
    """
    dataset_df = dataframe.copy()

    # Encoding the branch feature categories
    branch_maps = {
        "Offentligt": 1,
        "Erhverv": 2,
        "Privat": 3,
    }
    dataset_df["branch"] = (
        dataset_df["branch"].apply(lambda x: branch_maps.get(x)).astype("int8")
    )

    # Creating a new feature for consumption,
    # converting kilowatt-hour into megawatt-hour

    # dataset_df['consumption_mwh'] = dataset_df['consumption_kwh'] / 1000

    # Creating new columns based on the data present in the feature datetime_dk
    # dataset_df['hour_dk'] = dataset_df.datetime_dk.dt.hour
    # dataset_df['dayofmonth_dk'] = dataset_df.datetime_dk.dt.day
    # dataset_df['month_dk'] = dataset_df.datetime_dk.dt.month
    # dataset_df['year_dk'] = dataset_df.datetime_dk.dt.year
    # dataset_df['dayofweek_dk'] = dataset_df.datetime_dk.dt.dayofweek
    # dataset_df['dayofyear_dk'] = dataset_df.datetime_dk.dt.dayofyear
    # dataset_df['weekofyear'] = dataset_df.datetime_dk.dt.isocalendar().week
    # dataset_df['quarter_dk'] = dataset_df.datetime_dk.dt.quarter

    return dataset_df
