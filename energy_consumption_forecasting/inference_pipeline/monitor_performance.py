import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import validate_call
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_percentage_error,
)

from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.inference_pipeline.batch_data import (
    get_batch_data_from_hopsworks,
)
from energy_consumption_forecasting.inference_pipeline.utils import (
    get_gcs_bucket,
    read_blob_from_bucket,
    write_blob_to_bucket,
)
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.utils import get_env_var

logger = get_logger(name=Path(__file__).name)

import hopsworks


@log_exception(logger=logger)
@validate_call
def compute_performance_metrics(
    feature_view_ver: int = 1,
    feature_view_name: str = "denmark_energy_consumption_view",
    target_feature: str = "consumption_kwh",
) -> None:
    """
    This function computes performance metrics like MAPE and RMSPE using the
    predicted dataset and ground truth dataset, the metric result will be based on the
    datetime feature.

    Prediction dataset is downloaded from GCS bucket named cached_prediction, this is
    generated during the inference pipeline and ground truth is download from
    hopsworks feature store.

    Parameters
    ----------
    feature_view_ver: int, default=1
        The feature view version for ground truth dataset.

    feature_view_name: str, default="denmark_energy_consumption_view"
        The name of the feature view in the hopsworks feature store.

    target_feature: str, default="consumption_kwh"
        The name of the target feature in the dataset.
    """

    # Getting the bucket object from GCS and reading the cached prediction data
    logger.info("Connecting to the GCS bucket and reading the cached prediction data.")
    bucket = get_gcs_bucket()
    logger.info("Connection to the GCS bucket is established.")
    cached_prediction = read_blob_from_bucket(
        bucket=bucket,
        blob_name="cached_prediction.parquet",
    )

    if cached_prediction is None or len(cached_prediction) == 0:
        logger.info(
            "Cached prediction does not exist or data is missing, "
            "exiting the process."
        )
        return None

    logger.info("Successfully loaded cached prediction data.")

    # Connecting to hopsworks feature store and getting data with
    # similar date range of the prediction
    project = hopsworks.login(
        project=get_env_var(key="FEATURE_STORE_PROJECT_NAME"),
        api_key_value=get_env_var(key="FEATURE_STORE_API_KEY"),
    )
    feature_store = project.get_feature_store()
    logger.info(
        f'Connected to Hopsworks: Project Name "{project.name}" and '
        f'Project URL: "{project.get_url()}"'
    )

    pred_start_datetime = cached_prediction.index.get_level_values(
        level="datetime_dk"
    ).min()
    pred_end_datetime = cached_prediction.index.get_level_values(
        level="datetime_dk"
    ).max()
    logger.info(
        "Getting data from hopsworks feature store between "
        f'"{pred_start_datetime}" and "{pred_end_datetime}".'
    )
    _, ground_truth = get_batch_data_from_hopsworks(
        feature_store=feature_store,
        start_datetime=pred_start_datetime.to_timestamp(),
        # Adding 1 hour for less then condition in end_datetime
        end_datetime=(pred_end_datetime + 1).to_timestamp(),
        feature_view_ver=feature_view_ver,
        feature_view_name=feature_view_name,
        target_feature=target_feature,
    )

    if len(ground_truth) == 0:
        logger.info(
            "Ground truth data does not exist or data is missing, "
            "exiting the process."
        )
        return None

    gt_start_datetime = (
        ground_truth.index.get_level_values(level="datetime_dk").min().to_timestamp()
    )
    gt_end_datetime = (
        ground_truth.index.get_level_values(level="datetime_dk").max().to_timestamp()
    )
    logger.info(
        "Ground truth data received from hopsworks feature store, data is between "
        f'"{gt_start_datetime}" and "{gt_end_datetime}".'
    )

    logger.info("Successfully loaded ground truth data.")

    # Merging both the dataframe for computing
    logger.info("Computing performance metrics.")
    cached_prediction = cached_prediction.rename(
        columns={"consumption_kwh": "consumption_kwh_pred"}
    )
    ground_truth = ground_truth.rename(
        columns={"consumption_kwh": "consumption_kwh_gt"}
    )
    cached_prediction["consumption_kwh_gt"] = np.nan
    cached_prediction.update(ground_truth)

    # Computing MAPE and RMSPE performance metric between prediction and ground truth
    cached_prediction = cached_prediction.dropna(subset=["consumption_kwh_gt"])
    if len(cached_prediction) == 0:
        logger.info(
            "Ground truth data does not exist or data is missing, "
            "exiting the process."
        )
        return None

    mape = (
        cached_prediction.groupby("datetime_dk")
        .apply(
            lambda data: mean_absolute_percentage_error(
                y_true=data["consumption_kwh_gt"],
                y_pred=data["consumption_kwh_pred"],
                symmetric=False,
            )
        )
        .rename("mape")
        .to_frame()
    )

    rmspe = (
        cached_prediction.groupby("datetime_dk")
        .apply(
            lambda data: mean_squared_percentage_error(
                y_true=data["consumption_kwh_gt"],
                y_pred=data["consumption_kwh_pred"],
                square_root=True,
            )
        )
        .rename("rmspe")
        .to_frame()
    )

    metric_result = pd.merge(left=mape, right=rmspe, on="datetime_dk")
    logger.info("Performance metrics of MAPE and RMSPE successfully computed.")

    # Saving the metrics dataframe in the GCS bucket as a parquet file
    logger.info("Saving performance metrics dataframe in GCS bucket.")
    write_blob_to_bucket(
        bucket=bucket, blob_name="performance_metrics.parquet", data=metric_result
    )
    logger.info(
        'Performance metrics file "performance_metrics.parquet" saved in GCS bucket.'
    )

    # Saving the ground truth dataframe in the GCS bucket as a parquet file
    logger.info("Saving ground truth dataframe in GCS bucket.")
    ground_truth = ground_truth.rename(
        columns={"consumption_kwh_gt": "consumption_kwh"}
    )
    write_blob_to_bucket(
        bucket=bucket, blob_name="ground_truth.parquet", data=ground_truth
    )
    logger.info('Ground truth file "ground_truth.parquet" saved in GCS bucket.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--views_name",
        type=str,
        default="denmark_energy_consumption_view",
        help="Name of feature view within the feature group, needs to be in string format.",
    )

    parser.add_argument(
        "--views_ver",
        type=int,
        default=1,
        help="Version of feature view within the feature group, "
        "needs to be in integer format.",
    )

    parser.add_argument(
        "--target_feature",
        type=str,
        default="consumption_kwh",
        help="Name of target feature, needs to be in string format.",
    )

    args = parser.parse_args()

    compute_performance_metrics(
        feature_view_ver=args.views_ver,
        feature_view_name=args.views_name,
        target_feature=args.target_feature,
    )
