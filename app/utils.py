from typing import Dict, List

import pandas as pd


def build_dataframe(
    data_dict: Dict[str, List],
) -> pd.DataFrame:
    """
    This function builds a dataframe from the provided
    datetime and consumption arguments.

    Parameters
    ----------
    data_dict: Dict[str, List]
        A dictionary containing the column names as a key and the values as a list for
        that column.
        Note: One of the keys need to be a "datetime" with a list containing values
              with Unix epoch of hourly time format.

    Returns
    -------
    pd.DataFrame
        A dataframe containing column, a datetime column and another with some
        values durning that datetime.
    """

    dataframe = pd.DataFrame.from_dict(data=data_dict)

    # Converting hourly unix epoch into datetime format
    dataframe["datetime"] = pd.to_datetime(arg=dataframe["datetime"], unit="h")

    # Indexing and resampling the datetime feature in hourly frequency
    dataframe = dataframe.set_index(keys="datetime")
    dataframe = dataframe.resample(rule="H").asfreq()
    dataframe = dataframe.reset_index()

    return dataframe
