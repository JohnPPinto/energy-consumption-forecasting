from pathlib import Path
from typing import Tuple

import pandas as pd
from pydantic import validate_call
from sktime.split import temporal_train_test_split

import wandb
from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.training_pipeline.utils import init_wandb_run
from energy_consumption_forecasting.utils import get_env_var

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))

import hopsworks


def prepare_data(
    dataframe: pd.DataFrame,
    target_feature: str = "consumption_kwh",
    forecasting_horizon: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function prepares the dataset in train and test split for temporal dataframe.
    The test frame is based on the forecasting horizon.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe that needs to be prepared.

    target_feature: str, default="consumption_kwh"
        The target feature in the dataframe.

    forecasting_horizon: int, default=24
        The forecasting horizon for the test split size in hours, by default 24 hours.

    Returns
    -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Dataframe split in train and test: (y_train, y_test, X_train, X_test).
    """

    # Converting the datetime and setting the input columns to index
    dataframe["datetime_dk"] = pd.PeriodIndex(dataframe["datetime_dk"], freq="H")
    dataframe = dataframe.set_index(
        ["municipality_num", "branch", "datetime_dk"]
    ).sort_index()

    # Splitting the input and target features
    X = dataframe.drop(columns=[target_feature])
    y = dataframe[[target_feature]]

    # Creating a time-series train test split
    y_train, y_test, X_train, X_test = temporal_train_test_split(
        y=y.sort_index(),
        X=X.sort_index(),
        test_size=forecasting_horizon,
    )

    return y_train, y_test, X_train, X_test


@log_exception(logger=logger)
@validate_call
def load_prepared_dataset_from_feature_store(
    feature_view_name: str = "denmark_energy_consumption_view",
    feature_view_ver: int = 1,
    training_dataset_ver: int = 1,
    target_feature: str = "consumption_kwh",
    forecasting_horizon: int = 24,
):
    """
    This function loads the feature view from feature store and gets the training
    dataset from the feature view.\nAll the metadata from feature view is stored in
    the WandB artifact with the same name of feature view.

    Once the data is received, the dataset is prepared (splitted in train and test) and
    the metadata for prepaid process is logged as artifact in Wandb with
    name "train_split" and "test_split".

    Parameters
    ----------
    feature_view_name: str, default="denmark_energy_consumption_view"
        The name of the feature view in the hopsworks feature store.

    feature_view_ver: int, default=1
        The feature view version that needs to be loaded.

    training_dataset_ver: int, default=1
        The training dataset version within the feature view that needs to be downloaded.

    target_feature: str, default="consumption_kwh"
        The name of the target feature in the dataset.

    forecasting_horizon: int, default=24
        The forecasting horizon for the test split size in hours, by default 24 hours.

    Returns
    -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Dataframe splitted in train and test: (y_train, y_test, X_train, X_test).
    """

    # Connecting to the hopsworks feature store using the project API
    energy_project = hopsworks.login(
        project=get_env_var(key="FEATURE_STORE_PROJECT_NAME"),
        api_key_value=get_env_var(key="FEATURE_STORE_API_KEY"),
    )

    feature_store = energy_project.get_feature_store()

    logger.info(
        f'Connected to Hopsworks: Project Name "{energy_project.name}" and '
        f'Project URL: "{energy_project.get_url()}"'
    )

    # Getting the feature view metadata and training dataset from feature store
    # Creating a new artifact in wandb and storing the metadata in it.
    with init_wandb_run(
        run_name="get_training_dataset",
        job_type="load_feature_view",
        group="dataset",
    ) as run:

        logger.info(
            f"Details of the wandb run: run ID: {run.id}, run name: {run.name}, "
            f"project name: {run.project}, entity: {run.entity}, run group: "
            f"{run.group}, run save directory: {run.dir}, run URL: {run.url}"
        )

        feature_view = feature_store.get_feature_view(
            name=feature_view_name,
            version=feature_view_ver,
        )

        data, _ = feature_view.get_training_data(
            training_dataset_version=training_dataset_ver
        )

        metadata = feature_view.to_dict()
        metadata["query"] = metadata["query"].to_string()
        metadata["features"] = [i.name for i in metadata["features"]]
        metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
            feature_view=feature_view
        )
        metadata["feature_view_version"] = feature_view_ver
        metadata["training_dataset_version"] = training_dataset_ver

        run.log_artifact(
            wandb.Artifact(
                name=feature_view_name,
                type="feature_view",
                metadata=metadata,
            )
        )

        logger.info(f"Artifact {feature_view_name} has been logged successfully")

        run.finish()

    # Preparing the data by splitting it into train and test set
    # Logging the metadata for data preparation in wandb artifact.
    with init_wandb_run(
        run_name="train_test_split",
        job_type="prepare_dataset",
        group="dataset",
    ) as run:

        logger.info(
            f"Details of the wandb run: run ID: {run.id}, run name: {run.name}, "
            f"project name: {run.project}, entity: {run.entity}, run group: "
            f"{run.group}, run save directory: {run.dir}, run URL: {run.url}"
        )

        y_train, y_test, X_train, X_test = prepare_data(
            dataframe=data,
            target_feature=target_feature,
            forecasting_horizon=forecasting_horizon,
        )

        for i, (X, y) in enumerate(zip([X_train, X_test], [y_train, y_test])):

            metadata = {
                "dataset_time_interval": [
                    X.index.get_level_values(-1).min(),
                    X.index.get_level_values(-1).max(),
                ],
                "dataset_size": len(X),
                "total_municipality_num": len(X.index.get_level_values(0).unique()),
                "total_branch": len(X.index.get_level_values(1).unique()),
                "X_features": X.columns.to_list(),
                "y_features": y.columns.to_list(),
            }

            split = "train" if i == 0 else "test"
            run.log_artifact(
                wandb.Artifact(
                    name=f"{split}_split",
                    type="split_dataframe",
                    metadata=metadata,
                )
            )

            logger.info(f"Artifact {split}_split has been logged successfully")

        run.finish()

    return y_train, y_test, X_train, X_test
