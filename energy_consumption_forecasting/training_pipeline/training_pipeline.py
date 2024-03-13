import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import validate_call
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_percentage_error,
)
from sktime.utils.plotting import plot_series

import wandb
from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.training_pipeline.data_preprocessing import (
    load_prepared_dataset_from_feature_store,
)
from energy_consumption_forecasting.training_pipeline.model_builder import (
    build_lightgbm_model,
    build_naive_forecast_model,
)
from energy_consumption_forecasting.training_pipeline.utils import init_wandb_run
from energy_consumption_forecasting.utils import get_env_var, save_json_data

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))
DATA_DIRPATH = ROOT_DIRPATH / "data" / "assets" / "training"

import hopsworks


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame, fh: int = 24):
    """
    Train the forecast model using the given training dataset and forecast horizon.

    Parameters
    ----------
    model
        A sktime forecasting model or a forecasting pipeline.

    X_train: pd.DataFrame
        A pandas dataframe for training the model not containing the target feature.

    y_train: pd.DataFrame
        A pandas dataframe for training the model containing the target feature.

    Return
    ------
    model
        The model is been trained and returned back for performing prediction.
    """

    fh = np.arange(fh) + 1
    model.fit(y=y_train, X=X_train, fh=fh)

    return model


def perform_forecast(
    model, X_dataframe: pd.DataFrame, fh: int | None = None
) -> pd.DataFrame:
    """
    This function performs forecast on the provided model. The model should already
    be trained and the data provided should be different from the trained dataframe.

    Parameters
    ----------

    model
        A sktime forecasting model or a forecasting pipeline.

    X_dataframe: pd.DataFrame
        A pandas dataframe for performing forecast using the model,
        data should not contain the target feature.

    fh: int or None, default=None
        The forecasting horizon encoding the time stamps to forecast at.
        Should not be passed if has already been passed in model training.
        If has not been passed while model training, must be passed, not optional

    Returns
    -------
    pd.DataFrame
        Returns the predicted or forecasted dataframe from the point of forecast horizon
        with the same index as fh and same type as the data provided.
    """

    if fh is not None:
        fh = np.arange(fh) + 1

    return model.predict(X=X_dataframe, fh=fh)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    fh: int | None = None,
    save_plot: bool = True,
    model_name: str = "forecast_model",
    model_id_or_ver: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float | pd.DataFrame]]:
    """
    This function evaluates the model and calculates the error deviation using the MAPE
    and RMSPE performance metrics.

    Parameters
    ----------
    model
        A sktime forecasting model or a forecasting pipeline.

    X_test: pd.DataFrame
        A pandas dataframe for training the model not containing the target feature.

    y_train: pd.DataFrame
        A pandas dataframe for training the model containing the target feature.

    fh: int or None, default=None
        The forecasting horizon encoding the time stamps to forecast at.
        Should not be passed if has already been passed in model training.
        If has not been passed while model training, must be passed, not optional

    save_plot: bool, default=True
        Whether to save the plot locally and log it ti wandb.

    model_name: str, default="forecast_model"
        Model name to create a directory to save the plots locally

    model_id_or_ver: int, default=1
        A integer for model identification.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float | pd.DataFrame]]
        A prediction dataframe and dict containing the performance metrics result.
    """

    # Creating a directory to save plots
    save_dirpath = DATA_DIRPATH / f"{model_name}_{model_id_or_ver}"
    if not save_dirpath.exists():
        os.makedirs(save_dirpath)

    # Performing model
    y_pred = perform_forecast(model=model, X_dataframe=X_test, fh=fh)

    # Calculating quantitative performance metrics and adding it in a result dict.
    result_dict = {}
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred, symmetric=False)
    rmspe = mean_squared_percentage_error(
        y_true=y_test, y_pred=y_pred, square_root=True
    )
    result_dict["mape"] = mape
    result_dict["rmspe"] = rmspe

    # Grouping the data and evaluating for every branch in the hierarchy
    grouped_result_df = pd.DataFrame(
        columns=["municipality_num", "branch", "mape", "rmspe"]
    )
    y_test_group = y_test.reset_index(level=[0, 1]).groupby(
        ["municipality_num", "branch"]
    )
    y_pred_group = y_pred.reset_index(level=[0, 1]).groupby(
        ["municipality_num", "branch"]
    )

    for y_test_group, y_pred_group in zip(y_test_group, y_pred_group):
        (y_test_m_num, y_test_branch), y_test_data = y_test_group
        (y_pred_m_num, y_pred_branch), y_pred_data = y_pred_group

        assert (
            y_test_m_num == y_pred_m_num and y_test_branch == y_pred_branch
        ), "y_test and y_pred grouped data are not aligned"

        # Calculating quantitative performance metric on every branched data
        mape_group = mean_absolute_percentage_error(
            y_true=y_test_data, y_pred=y_pred_data, symmetric=False
        )
        rmspe_group = mean_squared_percentage_error(
            y_true=y_test_data, y_pred=y_pred_data, square_root=True
        )

        group_df = pd.DataFrame(
            {
                "municipality_num": [y_test_m_num],
                "branch": [y_test_branch],
                "mape": [mape_group],
                "rmspe": [rmspe_group],
            }
        )
        grouped_result_df = pd.concat([grouped_result_df, group_df], ignore_index=True)

        # Plotting actual and prediction results and saving the plot
        if save_plot:
            fig, axs = plot_series(
                y_test_data.consumption_kwh,
                y_pred_data.consumption_kwh,
                labels=["Actual - y_test", "Prediction - y_pred"],
                markers=["", ""],
            )
            fig.suptitle(
                f"Municipality Num: {y_test_m_num} - Branch: {y_test_branch}\n"
            )

            plot_filepath = (
                save_dirpath / f"pred_plot_{y_test_m_num}_{y_test_branch}.png"
            )
            plt.savefig(plot_filepath)
            plt.close(fig)

            wandb.log({plot_filepath.stem: wandb.Image(str(plot_filepath))})

    # Plotting and saving performance metrics result
    if save_plot:
        fig, axs = plt.subplots(2, 1, figsize=(25, 25))
        fig.suptitle(
            t="Quantitative Performance Metrics: MAPE and RMSPE\n\n",
            weight="bold",
            size=25,
        )

        for i, metrics in enumerate(["mape", "rmspe"]):
            sns.barplot(
                data=grouped_result_df,
                x="municipality_num",
                y=metrics,
                hue="branch",
                palette="Set1",
                ax=axs[i],
            )
            axs[i].set_xticks(axs[i].get_xticks())
            axs[i].set_xticklabels(
                grouped_result_df.municipality_num.unique(), rotation=90, size=12
            )
            axs[i].tick_params(axis="y", labelsize=12)
            axs[i].set_xlabel("Municipality Number\n", size=20)
            axs[i].set_ylabel(f"{metrics.upper()}", size=20)
            axs[i].legend(
                title="Branch",
                title_fontsize=15,
                fontsize=12,
            )
        plt.tight_layout()
        plot_filepath = save_dirpath / "metrics_result.png"
        plt.savefig(plot_filepath)
        plt.close(fig)

        wandb.log({plot_filepath.stem: wandb.Image(str(plot_filepath))})

        logger.info(
            f"All the plot are saved in the directory: {save_dirpath} "
            "and also logged in the WandB."
        )
        logger.info(
            "Prediction comparison plot are saved in format "
            '"pred_plot_<municipality_num>_<branch>.png" and metrics_result.png'
        )

    # Adding the municipality_num and branch grouped
    # error metrics dataframe in the result dict
    result_dict["grouped_result_df"] = grouped_result_df

    return y_pred, result_dict


def save_model_to_hopsworks(
    model_name: str,
    model_ver: int,
    model_filepath: Path,
    model_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Save the model within the hopsworks registry, by connecting to the hopsworks
    account, to connect the hopsworks account your project name and API key needs to be
    mentioned as a env variable.

    Parameters
    ----------
    model_name: str
        Name of the model in string format.

    model_ver: int
        Version of the model in integer format.

    model_filepath: pathlib.Path
        Locally saved model filepath in pathlib.Path format.

    model_metrics: Dict[str, float]
        Performance metrics to log with the model needs to be in dict format.

    Returns
    -------
    Dict[str, Any]
        Once the model is registered the hopsworks model registry generates a
        metadata, containing all the details in dict format.

    """
    # Login in into hopsworks and registering the model
    project = hopsworks.login(
        project=get_env_var(key="FEATURE_STORE_PROJECT_NAME"),
        api_key_value=get_env_var(key="FEATURE_STORE_API_KEY"),
    )
    logger.info(
        f'Connected to Hopsworks: Project Name "{project.name}" and '
        f'Project URL: "{project.get_url()}"'
    )

    # Uploading the model to the registry
    model_registry = project.get_model_registry()
    model_meta_obj = model_registry.python.create_model(
        name=model_name, version=model_ver, metrics=model_metrics
    )
    model_meta_obj.save(model_path=model_filepath)
    logger.info(
        f"Model: {model_name} version: {model_ver} is been registered in hopsworks"
    )

    return model_meta_obj.to_dict()


@log_exception(logger=logger)
@validate_call
def model_training_pipeline(
    fh: int = 24,
    summarize_period: List[int] = [24, 48, 72],
    feature_view_name: str = "denmark_energy_consumption_view",
    feature_view_ver: int = 1,
    training_dataset_ver: int = 1,
    target_feature: str = "consumption_kwh",
    save_plot: bool = True,
    model_type: Literal["naive", "lightgbm"] = "lightgbm",
    model_name: str = "forecast_model",
    model_id_or_ver: int = 1,
) -> Dict[str, Any]:
    """
    This function performs the model training pipeline, where it builds the specified
    model, trains the model and evaluates the model performance. Once the steps are
    completed the model is logged as a artifact in WandB and registered in the
    hopsworks model registry.

    Parameters
    ----------

    fh: int, default=24
        A period that indicates the forecast horizon while making prediction in Hours.

    summarize_period: List[int], default=[24, 48, 72]
        The period at which the window summarizer transformer will be applied and
        calculate the lag, mean and std.
        For eg. [24, 48, 72] indicate: lag of 72 period, mean and std of first
        24 period, then 48 period and last 72 period.
        Note: Period can be sort of duration, in the above example it is hours.

    feature_view_name: str, default="denmark_energy_consumption_view"
        The name of the feature view in the hopsworks feature store.

    feature_view_ver: int, default=1
        The feature view version that needs to be loaded.

    training_dataset_ver: int, default=1
        The training dataset version within the feature view that needs to be downloaded.

    target_feature: str, default="consumption_kwh"
        The name of the target feature in the dataset.

    save_plot: bool, default=True
        Save the generated plot locally and log it in WandB, takes in only boolean values.

    model_type: Literal["naive", "lightgbm"], default="lightgbm",
        Build, train and evaluate the model, currently supports only two models "naive" and
        "lightgbm" as model_type values.

    model_name: str, default="forecast_model"
        Name of the model for identifying and using the model to save it locally
        and log it on WandB and hopsworks.

    model_id_or_ver: int, default=1
        Version or ID of the model for identification purpose.

    Returns
    -------
    Dict[str, Any]
        A metadata dict is returned, this metadata dict is also been logged in the WandB
        and saved locally as a JSON file at the location: "./data/assets/training/
        <model_name>_<model_ver_or_id>/<model_name>_<model_ver_or_id>.json"
    """

    # Getting the training and testing dataset from feature store
    y_train, y_test, X_train, X_test = load_prepared_dataset_from_feature_store(
        feature_view_name=feature_view_name,
        feature_view_ver=feature_view_ver,
        training_dataset_ver=training_dataset_ver,
        target_feature=target_feature,
        forecasting_horizon=fh,
    )

    # Getting the datetime range of the training and testing dataset
    train_start_datetime = y_train.index.get_level_values("datetime_dk").min()
    train_end_datetime = y_train.index.get_level_values("datetime_dk").max()
    test_start_datetime = y_test.index.get_level_values("datetime_dk").min()
    test_end_datetime = y_test.index.get_level_values("datetime_dk").max()

    logger.info(
        f'Training dataset datetime range is from "{train_start_datetime}" to '
        f"{train_end_datetime}."
    )
    logger.info(
        f'Testing dataset datetime range is from "{test_start_datetime}" to '
        f"{test_end_datetime}."
    )

    with init_wandb_run(
        run_name=f"{model_name}_{model_id_or_ver}",
        add_timestamp_to_run_name=True,
        job_type="model_training",
        group="model",
    ) as run:

        # Building and training model as per the provided model argument
        if model_type == "naive":
            forecast_model = build_naive_forecast_model(seasonal_periodicity=fh)
            logger.info("Naive forecaster model is been build and ready for training")

        elif model_type == "lightgbm":
            # Getting the saved hyperparameter file from the wandb artifact
            model_params_artifact = run.use_artifact(
                artifact_or_name="best_config:latest",
                type="model",
            )
            artifact_dir = model_params_artifact.download()
            model_params_path = Path(artifact_dir) / "best_config.json"
            with open(model_params_path) as file:
                model_params = json.load(file)

            # Updating the WandB config with the hyperparameters
            run.config.update(model_params)

            forecast_model = build_lightgbm_model(
                summarize_period=summarize_period,
                model_params=model_params,
            )
            logger.info("LightGBM model is been build and ready for training")

        forecast_model = train_model(
            model=forecast_model, X_train=X_train, y_train=y_train, fh=fh
        )

        y_pred, metrics_dict = evaluate_model(
            model=forecast_model,
            X_test=X_test,
            y_test=y_test,
            fh=fh,
            save_plot=save_plot,
            model_name=model_name,
            model_id_or_ver=model_id_or_ver,
        )
        metrics_df = metrics_dict.pop("grouped_result_df")

        pred_start_datetime = y_pred.index.get_level_values("datetime_dk").min()
        pred_end_datetime = y_pred.index.get_level_values("datetime_dk").max()
        logger.info(
            f'Predicted dataframe datetime range is from "{pred_start_datetime}" to '
            f"{pred_end_datetime}."
        )
        for k, v in metrics_dict.items():
            logger.info(f"Baseline metric dict: {k}: {v}")
        wandb.log({"metrics": {f"{model_name}_{model_id_or_ver}": metrics_dict}})
        wandb.log(
            {
                f"metric/{model_type}/{model_name}_{model_id_or_ver}": wandb.Table(
                    dataframe=metrics_df
                )
            }
        )

        # Saving the model locally as a pickle file
        save_dirpath = DATA_DIRPATH / f"{model_name}_{model_id_or_ver}"
        if not save_dirpath.exists():
            os.makedirs(save_dirpath)
        save_model_filepath = save_dirpath / f"{model_name}_{model_id_or_ver}.pkl"
        joblib.dump(value=forecast_model, filename=save_model_filepath)
        logger.info(
            f"Model pickle file is been saved locally at: {save_model_filepath}"
        )

        # Creating a wandb artifact of model and logging it in wandb
        # and also saving the model in the hopsworks model registry
        metadata = {
            "experiment": {
                "fh": fh,
                "summarize_period": summarize_period,
                "feature_view_name": feature_view_name,
                "feature_view_ver": feature_view_ver,
                "training_dataset_ver": training_dataset_ver,
                "target_feature": target_feature,
                "model_type": model_type,
                "model_name": model_name,
                "model_id_or_ver": model_id_or_ver,
                "training_start_datetime": train_start_datetime.to_timestamp().isoformat(),
                "training_end_datetime": train_end_datetime.to_timestamp().isoformat(),
                "testing_start_datetime": test_start_datetime.to_timestamp().isoformat(),
                "testing_end_datetime": test_end_datetime.to_timestamp().isoformat(),
                "prediction_start_datetime": pred_start_datetime.to_timestamp().isoformat(),
                "prediction_end_datetime": pred_end_datetime.to_timestamp().isoformat(),
            },
            "result": {"metrics": metrics_dict},
        }

        metadata["model_registry"] = save_model_to_hopsworks(
            model_name=model_name,
            model_ver=model_id_or_ver,
            model_filepath=save_model_filepath,
            model_metrics=metrics_dict,
        )

        artifact = wandb.Artifact(
            name=f"{model_name}_{model_id_or_ver}", type="model", metadata=metadata
        )
        artifact.add_file(local_path=save_model_filepath)
        run.log_artifact(artifact)

        metadata_filepath = save_dirpath / f"{model_name}_{model_id_or_ver}.json"
        save_json_data(data=metadata, filepath=metadata_filepath)
        logger.info(
            f"Training metadata file is been saved locally at: {metadata_filepath}"
        )

        run.finish()

    return metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fh",
        type=int,
        default=24,
        help="Forecasting horizon period, needs to be in integer format.",
    )

    parser.add_argument(
        "--summarize_period",
        nargs="+",
        default=[24, 48, 72],
        help="Period to summarize the target feature, "
        "format needs to be in integer and multiple values are accepted.",
    )

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
        "--dataset_ver",
        type=int,
        default=1,
        help="Version of training dataset within the feature view, "
        "needs to be in integer format.",
    )

    parser.add_argument(
        "--target_feature",
        type=str,
        default="consumption_kwh",
        help="Name of target feature, needs to be in string format.",
    )

    parser.add_argument(
        "--save_plot",
        type=bool,
        default=True,
        help="Whether to save plot while training and evaluating the model on dataset, "
        'only boolean values accepted either "True" or "False"',
    )

    parser.add_argument(
        "--model_type",
        type=str.lower,
        default="lightgbm",
        choices=["naive", "lightgbm"],
        help="Which type of model needs to be build, trained and forecasted, "
        'currently accepts only "naive" and "lightgbm" models.',
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="forecast_model",
        help="Name of the model for this training experiment, needs to be in string format.",
    )

    parser.add_argument(
        "--model_ver",
        type=int,
        default=1,
        help="Model version or id for this training experiment, "
        "needs to be in integer format.",
    )

    args = parser.parse_args()

    model_training_pipeline(
        fh=args.fh,
        summarize_period=args.summarize_period,
        feature_view_name=args.views_name,
        feature_view_ver=args.views_ver,
        training_dataset_ver=args.dataset_ver,
        target_feature=args.target_feature,
        save_plot=args.save_plot,
        model_type=args.model_type,
        model_name=args.model_name,
        model_id_or_ver=args.model_ver,
    )
