import os
from pathlib import Path
from typing import Optional

import hopsworks
import pandas as pd
from great_expectations.core import ExpectationSuite
from hsfs.feature_group import FeatureGroup
from pydantic import validate_call

from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.utils import get_env_var

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))
PROCESSED_DATA_DIRPATH = ROOT_DIRPATH / "data" / "processed_data"


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def loading_data_to_hopsworks(
    dataframe: pd.DataFrame,
    generated_expectation_suite: ExpectationSuite,
    hopsworks_feature_group_name: str = "denmark_energy_consumption_group",
    hopsworks_feature_group_version: int = 1,
    save_data_offline_dirpath: Optional[str | Path] = PROCESSED_DATA_DIRPATH,
) -> FeatureGroup | Path:
    """
    This function loads the provided dataframe in the hopsworks feature store.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas dataframe that needs to be uploaded in the feature store.

    generated_expectation_suite: ExpectationSuite
        A instance of the ExpectationSuite with predefined configurations for
        performing data validation while uploading the data in the feature store.

    hopsworks_feature_group_name: str, default='denmark_energy_consumption_group'
        A name in string type to set the feature group name.

    hopsworks_feature_group_version: int, default=1
        A version number in int type to set the feature group version.

    save_data_offline_dir: str or Path or None, default='./data/processed_data/'
        A directory path to save the dataframe in the directory, filename will be auto
        generated using the project name and dataset starting and ending datetime,
        the saved dataset will be in a CSV file format.

    Returns
    -------
    hsfs.feature_group.FeatureGroup or pathlib.Path
        Returns the metadata of the created and updated feature group, including all the
        details of the dataframe.
        OR returns metadata and filepath of the dataframe saved in local directory, if
        save_data_offline_dirpath argument is been provided.
    """

    # Connecting to the hopsworks feature store using the project API
    energy_feature_project = hopsworks.login(
        project=get_env_var(key="FEATURE_STORE_PROJECT_NAME"),
        api_key_value=get_env_var(key="FEATURE_STORE_API_KEY"),
    )

    logger.info(
        f'Connected to Hopsworks: Project Name "{energy_feature_project.name}" and '
        f'Project URL: "{energy_feature_project.get_url()}"'
    )

    hopsworks_feature_store = energy_feature_project.get_feature_store()

    # Creating a feature group within the hopsworks project
    energy_feature_group = hopsworks_feature_store.get_or_create_feature_group(
        name=hopsworks_feature_group_name,
        version=hopsworks_feature_group_version,
        description=(
            "Denmark's energy consumption data for industry, private and "
            "public categories per municipality number, data is provided on hourly "
            "base with an approx 15 days of update frequency."
        ),
        online_enabled=False,
        primary_key=["municipality_num", "branch"],
        expectation_suite=generated_expectation_suite,
        event_time="datetime_dk",
    )

    # Uploading the data using the above created feature group
    energy_feature_group.insert(
        features=dataframe,
        overwrite=False,
        write_options={"wait_for_job": True},
    )

    # Updating features metadata for the dataframe in the features group
    features_metadata = [
        {
            "name": "datetime_dk",
            "description": (
                "A date and time (interval), shown in Danish time zone. "
                "The interval between data is of an hourly difference."
            ),
        },
        {
            "name": "municipality_num",
            "description": (
                "Each of the 98 Danish municipalities has a unique number, "
                "ranging from 101 Copenhagen to 860 Hj\u00f8rring."
            ),
        },
        {
            "name": "branch",
            "description": (
                "All measurement id's related to a CVR number are labeled as 'Erhverv'"
                "except those that belong to 'Office, admin, which are labeled "
                "'Offentligt'. All other measurement id's are labeled 'Privat'."
            ),
        },
        {
            "name": "consumption_kwh",
            "description": (
                "Total energy consumption in kilowatt-hour(kWh), "
                "the value should be greater than 0"
            ),
        },
    ]

    for data in features_metadata:
        energy_feature_group.update_feature_description(
            feature_name=data.get("name"),
            description=data.get("description"),
        )

    # Updating statistic configuration for the feature group
    energy_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }

    # Calling, updating and computing the feature group statistic
    energy_feature_group.update_statistics_config()
    energy_feature_group.compute_statistics()

    logger.info(
        "Data and Metadata has been uploaded to the feature store "
        f'group: "{energy_feature_group.name}"'
    )

    # Saving the dataframe in the local data directory
    if save_data_offline_dirpath is not None:

        # Converting the string into a Path datatype
        if isinstance(save_data_offline_dirpath, str):
            save_data_offline_dirpath = Path(save_data_offline_dirpath)

        # Checking whether the path directory exist or not
        if not save_data_offline_dirpath.is_dir():
            os.makedirs(save_data_offline_dirpath)

        # Getting the start and end date of the dataframe
        start_date = (
            str(dataframe.datetime_dk.iloc[0]).replace(" ", "T").replace(":", "-")
        )
        end_date = (
            str(dataframe.datetime_dk.iloc[-1]).replace(" ", "T").replace(":", "-")
        )

        filepath = save_data_offline_dirpath / (
            f"{energy_feature_group.name}_v{hopsworks_feature_group_version}_"
            f"{start_date}_{end_date}.csv"
        )

        dataframe.to_csv(path_or_buf=filepath, index=False)
        logger.info(f'Dataset has been saved in csv file at "{filepath}".')

        return energy_feature_group, filepath

    return energy_feature_group
