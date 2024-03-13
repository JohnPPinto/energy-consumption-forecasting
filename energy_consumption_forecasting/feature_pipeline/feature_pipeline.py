import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import validate_call

from energy_consumption_forecasting.exceptions import log_exception
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
@validate_call
def run_feature_pipeline(
    start_date_time: datetime.datetime,
    end_date_time: datetime.datetime,
    rename_features: Dict[Any, Any],
    drop_features: Optional[List[Any]] = ["HourUTC"],
    check_features_duplicates: List[Any] = [
        "municipality_num",
        "branch",
        "datetime_dk",
    ],
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

    rename_features: Dict[Any, Any]
        A dict containing existing column names as keys and new column names as values.

    drop_features: Optional[List[Any]], default=["HourUTC"]
        A list containing column names for dropping it from the DataFrame.

    check_features_duplicate: List[Any], default=["municipality_num", "branch", "datetime_dk"]
        A list containing column names for checking duplication and dropping the rows.

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

    dataframe = data_transformation.rename_features(
        dataframe=dataframe,
        rename_columns_dict=rename_features,
    )
    dataframe = data_transformation.casting_features(dataframe=dataframe)
    dataframe = data_transformation.feature_engineering(dataframe=dataframe)
    dataframe = data_transformation.clean_dataframe(
        dataframe=dataframe,
        drop_columns=drop_features,
        check_columns_duplicates=check_features_duplicates,
    )

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

    class ParseKVAction(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Sequence[Any] | None,
            option_string: str | None = None,
        ) -> None:
            setattr(namespace, self.dest, dict())
            for i in values:
                try:
                    key, value = i.split("=")
                    getattr(namespace, self.dest)[key] = value
                except ValueError as e:
                    message = f"\nTraceback: {e}"
                    message += f"\nError on {e}, format should be in 'key=value."
                    raise argparse.ArgumentError(self, str(message))

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
        default="2023-12-31",
        help="Ending date for extraction in format: YYYY-MM-DDTHH:MM:SS",
    )

    parser.add_argument(
        "-r",
        "--rename_features",
        nargs="+",
        action=ParseKVAction,
        default={
            "HourDK": "datetime_dk",
            "MunicipalityNo": "municipality_num",
            "Branche": "branch",
            "ConsumptionkWh": "consumption_kwh",
        },
        help="Rename the features to transform the dataframe, "
        "format needs to be in CURRENT_NAME=NEW_NAME ...",
        metavar="CURRENT_NAME=NEW_NAME",
    )

    parser.add_argument(
        "-d",
        "--drop_features",
        nargs="+",
        default=["HourUTC"],
        help="Drop the features to transform the dataframe, "
        "format needs to be in NAME_1 NAME_2 NAME_3 ...",
    )

    parser.add_argument(
        "-c",
        "--check_duplicates",
        nargs="+",
        default=["municipality_num", "branch", "datetime_dk"],
        help="Provide feature names to check for duplicate data and "
        "clean the dataframe, format needs to be in NAME_1 NAME_2 NAME_3 ...",
    )

    args = parser.parse_args()

    _, filepath = run_feature_pipeline(
        start_date_time=args.start_datetime,
        end_date_time=args.end_datetime,
        rename_features=args.rename_features,
        drop_features=args.drop_features,
        check_features_duplicates=args.check_duplicates,
    )

    print(f"\nLocally saved metadata filepath: {filepath}")
