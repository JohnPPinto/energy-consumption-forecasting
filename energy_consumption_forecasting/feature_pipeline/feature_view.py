import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import validate_call

from energy_consumption_forecasting.exceptions import (
    CustomExceptionMessage,
    log_exception,
)
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.utils import get_env_var, save_json_data

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))
DATA_DIRPATH = ROOT_DIRPATH / "data" / "processed_data"

import hopsworks


@log_exception(logger=logger)
@validate_call
def create_feature_view(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    feature_group_version: int = 1,
    feature_group_name: str = "denmark_energy_consumption_group",
    feature_views_name: str = "denmark_energy_consumption_view",
    feature_views_description: str = "Denmark's energy consumption forecasting model training view",
) -> Tuple[Dict[Any, Any], Path]:
    """
    This function creates a new feature view and training dataset on provided
    datetime and feature group data.

    Parameters
    ----------
    start_datetime: datetime.datetime
        A starting date and time for extracting the dataframe from feature group
        in datatype of datetime.datetime.

    end_datetime: datetime.datetime
        A ending date and time for extracting the dataframe from feature group
        in datatype of datetime.datetime.

    feature_group_version: int, default=1
         A version number in int type for getting the specific feature group.

    feature_group_name: str, default="denmark_energy_consumption_group"
        A feature group name in string type for getting the specific feature group.

    feature_views_name: str, default="denmark_energy_consumption_view"
        A feature view name in string type for naming the newly created view.

    feature_views_description: str, default="Denmark's energy consumption forecasting model training view"
        A feature view description in string format for setting the description of the
        newly created view.

    Returns
    -------
    Dict and pathlib.Path
        After creating the feature view and saving the metadata as a JSON file in the
        local directory, the function returns the metadata as a Dict object and the
        file path of the json file.
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

    # Deleting old feature views because currently using free tier service of hopsworks
    # In free tier there is a limited options for creating views so replacing every time
    # a new view needs to be created.
    try:
        energy_feature_views = feature_store.get_feature_views(name=feature_views_name)
    except Exception as e:
        print(CustomExceptionMessage(exception_msg=e))
        logger.info(
            "Feature store could not get the feature view: "
            f'"{feature_views_name}" in the feature store.'
        )

        energy_feature_views = []

    if len(energy_feature_views) > 0:
        for view in energy_feature_views:
            try:
                view.delete_all_training_datasets()
            except Exception as e:
                print(CustomExceptionMessage(e))

            try:
                view.delete()
            except Exception as e:
                print(CustomExceptionMessage(e))

            logger.info(
                f'Deleting feature view for name "{feature_views_name}": {view}'
            )

    # Creating feature view from the provided feature group
    energy_feature_group = feature_store.get_feature_group(
        name=feature_group_name,
        version=feature_group_version,
    )

    dataframe_query = energy_feature_group.select_all()

    energy_feature_view = feature_store.create_feature_view(
        name=feature_views_name,
        query=dataframe_query,
        description=feature_views_description,
    )

    # Creating a training dataset in the feature views
    logger.info(
        f'Creating a training dataset in the feature view: "{feature_views_name}" '
        f"between the start date: {start_datetime} and end date: {end_datetime}"
    )

    energy_feature_view.create_training_data(
        description=f"Training dataset between {start_datetime} and {end_datetime}",
        data_format="csv",
        start_time=start_datetime,
        end_time=end_datetime,
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    # Saving the metadata generated while creating the feature view
    energy_feature_view_metadata = energy_feature_view.json()
    energy_feature_view_metadata = json.loads(
        energy_feature_view_metadata.replace('\\"', '"')
        .replace('\\\\"', '"')
        .replace('"{', "{")
        .replace('}"', "}")
    )

    json_filepath = DATA_DIRPATH / f"{feature_views_name}_v1_metadata.json"
    save_json_data(data=energy_feature_view_metadata, filepath=json_filepath)

    logger.info(
        f"Feature view: '{feature_views_name}' and training dataset is been created and "
        f"the metadata is saved in the local directory as a JSON file: {json_filepath}."
    )

    return energy_feature_view_metadata, json_filepath


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--start_datetime",
        type=datetime.datetime.fromisoformat,
        default="2021-01-01",
        help="Starting date for extraction in format: YYYY-MM-DDTHH:MM:SS",
    )

    parser.add_argument(
        "-e",
        "--end_datetime",
        type=datetime.datetime.fromisoformat,
        default="2024-01-01",
        help="Ending date for extraction in format: YYYY-MM-DDTHH:MM:SS",
    )

    parser.add_argument(
        "--group_version",
        type=int,
        default=1,
        help="Feature group version, needs to be in integer format.",
    )

    parser.add_argument(
        "--group_name",
        type=str,
        default="denmark_energy_consumption_group",
        help="Feature group name, needs to be in string format.",
    )

    parser.add_argument(
        "--views_name",
        type=str,
        default="denmark_energy_consumption_view",
        help="Feature view name within the feature group, needs to be in string format.",
    )

    parser.add_argument(
        "--views_desc",
        type=str,
        default="Denmark's energy consumption forecasting model training view",
        help="Description for the feature view, needs to be in string format.",
    )

    args = parser.parse_args()

    _, filepath = create_feature_view(
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        feature_group_version=args.group_version,
        feature_group_name=args.group_name,
        feature_views_name=args.views_name,
        feature_views_description=args.views_desc,
    )

    print(f"\nLocally saved feature view metadata, filepath: {filepath}")
