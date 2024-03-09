from datetime import datetime
from typing import Tuple

import pandas as pd
from hsfs.feature_store import FeatureStore


def get_batch_data_from_hopsworks(
    feature_store: FeatureStore,
    start_datetime: datetime,
    end_datetime: datetime,
    feature_view_ver: int = 1,
    feature_view_name: str = "denmark_energy_consumption_view",
    target_feature: str = "consumption_kwh",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This functions get the batch data from hopsworks feature view and process the batch
    data into the sktime data format for hierarchical dataset and then returns the
    input and target dataset.

    Parameters
    ----------
    feature_store: FeatureStore
        A instance of FeatureStore object that is connected to the hopsworks feature
        store and the project in which the data is stored.

    start_datetime: datetime.datetime
        A starting date and time for extracting the batch data
        from hopsworks feature view.

    end_datetime: datetime.datetime
        A ending date and time for extracting the batch data
        from hopsworks feature view.

    feature_view_ver: int, default=1
        The feature view version that needs to be loaded.

    feature_view_name: str, default="denmark_energy_consumption_view"
        The name of the feature view in the hopsworks feature store.

    target_feature: str, default="consumption_kwh"
        The name of the target feature in the dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
            Dataframe split in Input and Target sets as per sktime format: X, y.
    """

    # Getting the feature views where the data is been stored for batch processing
    feature_view = feature_store.get_feature_view(
        name=feature_view_name, version=feature_view_ver
    )

    batch_data = feature_view.get_batch_data(
        start_time=start_datetime, end_time=end_datetime
    )

    # Processing the data into sktime format:
    # X: Containing all the hierarchical features as multi-index format
    # y: Along with the multi-index features in X, target feature data needs to be present
    batch_data["datetime_dk"] = pd.PeriodIndex(batch_data["datetime_dk"], freq="H")
    batch_data = batch_data.set_index(
        ["municipality_num", "branch", "datetime_dk"]
    ).sort_index()

    # Splitting the input and target features
    X = batch_data.drop(columns=[target_feature])
    y = batch_data[[target_feature]]

    return X, y
