from typing import Dict, List

import gcsfs
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend import schemas
from backend.config import get_settings

gcs_filesystem = gcsfs.GCSFileSystem(
    project=get_settings().GCP_PROJECT,
    token=get_settings().GCP_SERVICE_ACCOUNT_FILE,
)

api_router = APIRouter()


@api_router.get(
    path="/health",
    response_model=schemas.HealthCheck,
    status_code=status.HTTP_200_OK,
)
def health() -> Dict:
    """
    This endpoint creates a health check endpoint.
    """

    health_check_data = schemas.HealthCheck(
        name=get_settings().PROJECT_NAME,
        api_version=get_settings().API_VERSION,
        status="OK",
    )

    return health_check_data.model_dump()


@api_router.get(
    path="/municipality_number_values",
    response_model=schemas.UniqueMunicipalityNumber,
    status_code=status.HTTP_200_OK,
)
def municipality_number_values() -> Dict[str, List]:
    """
    This endpoint gets the unique municipality number from the stored data in GCS.
    """

    # Connecting to GCS filesystem and downloading the data
    input_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/input.parquet",
        filesystem=gcs_filesystem,
    )

    unique_municipality_number = list(input_df.index.unique(level="municipality_num"))

    return {"values": unique_municipality_number}


@api_router.get(
    path="/branch_values",
    response_model=schemas.UniqueBranch,
    status_code=status.HTTP_200_OK,
)
def branch_values() -> Dict[str, List]:
    """
    This endpoint gets the unique branch values from the stored data in GCS.
    """

    # Connecting to GCS filesystem and downloading the data
    input_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/input.parquet",
        filesystem=gcs_filesystem,
    )

    unique_branch = list(input_df.index.unique(level="branch"))

    return {"values": unique_branch}


@api_router.get(
    path="/prediction/{municipality_number}/{branch}",
    response_model=schemas.PredictionResults,
    status_code=status.HTTP_200_OK,
)
async def get_prediction(municipality_number: int, branch: int) -> Dict[str, List]:
    """
    This endpoint gets the forecasted prediction for the provided municipality number
    and branch from the stored data in GCS.
    """

    # Connecting to GCS filesystem and downloading the data
    historical_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/target.parquet",
        filesystem=gcs_filesystem,
    )
    prediction_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/prediction.parquet",
        filesystem=gcs_filesystem,
    )

    # Filtering the dataframe with the provided municipality number and brach values
    try:
        historical_df = historical_df.xs(
            key=(municipality_number, branch), level=["municipality_num", "branch"]
        )
        prediction_df = prediction_df.xs(
            key=(municipality_number, branch), level=["municipality_num", "branch"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Data not found for the provided municipality number: "
                f"{municipality_number} and branch: {branch}.\nException "
                f"error detail as follows:\n\t{e}"
            ),
        )

    if len(historical_df) == 0 or len(prediction_df) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Data not found for the provided municipality number: "
                f"{municipality_number} and branch: {branch}."
            ),
        )

    # Returning the latest data that is been observed
    # historical_df = historical_df.sort_index().tail(24 * 31)

    # Returning the datetime and consumption for both the dataframe in list format
    historical_datetime = (
        historical_df.index.get_level_values("datetime_dk").astype("int64").to_list()
    )
    historical_consumption = historical_df.consumption_kwh.to_list()
    prediction_datetime = (
        prediction_df.index.get_level_values("datetime_dk").astype("int64").to_list()
    )
    prediction_consumption = prediction_df.consumption_kwh.to_list()

    result = {
        "historical_datetime": historical_datetime,
        "historical_consumption": historical_consumption,
        "prediction_datetime": prediction_datetime,
        "prediction_consumption": prediction_consumption,
    }

    return result


@api_router.get(
    path="/monitor/metrics",
    response_model=schemas.MonitorMetrics,
    status_code=status.HTTP_200_OK,
)
async def get_metrics() -> Dict[str, List]:
    """
    This endpoint get the performance metrics stored in the GCS.
    """

    # Connecting to GCS filesystem and downloading the data
    metrics_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/performance_metrics.parquet",
        filesystem=gcs_filesystem,
    )

    if len(metrics_df) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Data not found or either performance metrics has not been generated."
            ),
        )

    datetime = metrics_df.index.astype("int64").to_list()
    mape = metrics_df.mape.to_list()
    rmspe = metrics_df.rmspe.to_list()

    result = {"datetime": datetime, "mape": mape, "rmspe": rmspe}

    return result


@api_router.get(
    path="/monitor/prediction/{municipality_number}/{branch}",
    response_model=schemas.MonitorPrediction,
    status_code=status.HTTP_200_OK,
)
async def get_cached_prediction(
    municipality_number: int, branch: int
) -> Dict[str, List]:
    """
    This endpoint gets the forecasted cached prediction and its ground truth dataframe
    from GCS to monitor the prediction performance of the model by comparing
    it with the ground truth.
    """

    # Connecting to GCS filesystem and downloading the data
    ground_truth_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/ground_truth.parquet",
        filesystem=gcs_filesystem,
    )
    cached_prediction_df = pd.read_parquet(
        path=f"{get_settings().GCP_BUCKET_NAME}/cached_prediction.parquet",
        filesystem=gcs_filesystem,
    )

    # Filtering the dataframe with the provided municipality number and brach values
    try:
        ground_truth_df = ground_truth_df.xs(
            key=(municipality_number, branch), level=["municipality_num", "branch"]
        )
        cached_prediction_df = cached_prediction_df.xs(
            key=(municipality_number, branch), level=["municipality_num", "branch"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Data not found for the provided municipality number: "
                f"{municipality_number} and branch: {branch}.\nException "
                f"error detail as follows:\n\t{e}"
            ),
        )

    if len(ground_truth_df) == 0 or len(cached_prediction_df) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Data not found for the provided municipality number: "
                f"{municipality_number} and branch: {branch}."
            ),
        )

    # Returning the datetime and consumption for both the dataframe in list format
    ground_truth_datetime = (
        ground_truth_df.index.get_level_values("datetime_dk").astype("int64").to_list()
    )
    ground_truth_consumption = ground_truth_df.consumption_kwh.to_list()
    cached_prediction_datetime = (
        cached_prediction_df.index.get_level_values("datetime_dk")
        .astype("int64")
        .to_list()
    )
    cached_prediction_consumption = cached_prediction_df.consumption_kwh.to_list()

    result = {
        "ground_truth_datetime": ground_truth_datetime,
        "ground_truth_consumption": ground_truth_consumption,
        "cached_prediction_datetime": cached_prediction_datetime,
        "cached_prediction_consumption": cached_prediction_consumption,
    }

    return result
