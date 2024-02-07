import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import validate_call

from energy_consumption_forecasting.exception import log_exception
from energy_consumption_forecasting.feature_pipeline import (
    data_extraction,
    data_loading,
    data_transformation,
    data_validation,
)
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.utils import get_env_var, save_json_data

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))
DATA_DIRPATH = ROOT_DIRPATH / "data" / "processed_data"


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def run_feature_pipeline(
    start_date_time: datetime.datetime,
    end_date_time: datetime.datetime,
    drop_features: Optional[List[Any]],
    rename_features: Dict[Any, Any],
) -> Tuple[Dict[Any, Any], Path]:
    """
    This functions runs the feature pipeline process i.e. ETL - Extract, Transform and
    Load.\n
    Data is extracted using the API, transformation is been done using the pandas
    library and data is been validated and loaded using great expectation and hopsworks.

    Parameters
    ----------
    start_date_time: datetime.datetime
        A starting date and time for extracting the data in datatype of
        datetime.datetime.

    end_date_time: datetime.datetime
        A ending date and time for extracting the data in datatype of
        datetime.datetime.

    drop_features: Optional[List[Any]]
        A list containing column names for dropping it from the DataFrame.

    rename_features: Dict[Any, Any]
        A dict containing existing column names as keys and new column names as values.

    Returns
    -------
    dict and pathlib.Path
        Returns a json metadata of the feature store in dict format and filepath of the
        JSON file.
    """

    # Extracting the dataset
    logger.info("Starting dataset extraction process.")

    dataframe, _, _, _ = data_extraction.extract_dataset_from_api(
        start_date_time=start_date_time,
        end_date_time=end_date_time,
    )

    logger.info("Data extraction process is successfully completed.\n")

    # Transforming the dataframe
    logger.info("Starting dataframe transformation process.")

    dataframe = data_transformation.clean_dataframe(
        dataframe=dataframe,
        drop_columns=drop_features,
    )
    dataframe = data_transformation.rename_features(
        dataframe=dataframe,
        rename_columns_dict=rename_features,
    )
    dataframe = data_transformation.casting_features(dataframe=dataframe)
    dataframe = data_transformation.feature_engineering(dataframe=dataframe)

    logger.info("Data transformation process is successfully completed.\n")

    # Building the dataframe validation using the great expectation suite
    logger.info(
        "Starting the process to build the dataframe validation expectation suite."
    )

    generated_expectation_suite = data_validation.generate_great_expectation_suite()

    logger.info(
        "DataFrame validation expectation suite is been successfully generated.\n"
    )

    # Loading the dataframe into the feature store
    logger.info("Starting dataframe loading process.")

    feature_store_metadata, csv_filepath = data_loading.loading_data_to_hopsworks(
        dataframe=dataframe,
        generated_expectation_suite=generated_expectation_suite,
    )

    logger.info("Data loading process is successfully completed.\n")

    # Getting and cleaning the metadata and converting it into a JSON format
    feature_store_metadata = feature_store_metadata.json()
    feature_store_metadata = json.loads(
        feature_store_metadata.replace('\\"', '"')
        .replace('\\\\"', '"')
        .replace('"{', "{")
        .replace('}"', "}")
    )

    # Adding data extraction start and end datetime in metadata
    feature_store_metadata["data_extraction_start_datetime"] = str(start_date_time)
    feature_store_metadata["data_extraction_end_datetime"] = str(
        end_date_time + datetime.timedelta(days=1)
    )

    # Saving the provided feature store metadata in a local directory as a json file
    csv_filepath = csv_filepath.name.split("_")[:4]
    json_filepath = DATA_DIRPATH / f"{'_'.join(csv_filepath)}_metadata.json"
    save_json_data(data=feature_store_metadata, filepath=json_filepath)

    logger.info("Feature pipeline process is completed.")

    return feature_store_metadata, json_filepath


if __name__ == "__main__":

    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    rename_col = {
        "HourDK": "datetime_dk",
        "MunicipalityNo": "municipality_num",
        "Branche": "branch",
        "ConsumptionkWh": "consumption_kwh",
    }
    _, filepath = run_feature_pipeline(
        start_date_time=start_date,
        end_date_time=end_date,
        drop_features=["HourUTC"],
        rename_features=rename_col,
    )

    print(f"\nLocally saved metadata filepath: {filepath}")