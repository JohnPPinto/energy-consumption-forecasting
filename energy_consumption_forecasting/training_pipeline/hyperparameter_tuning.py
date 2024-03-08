import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import wandb
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
from pydantic import validate_call
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from energy_consumption_forecasting.exceptions import log_exception
from energy_consumption_forecasting.logger import get_logger
from energy_consumption_forecasting.training_pipeline.data_preprocessing import (
    load_prepared_dataset_from_feature_store,
)
from energy_consumption_forecasting.training_pipeline.model_builder import (
    build_lightgbm_model,
)
from energy_consumption_forecasting.training_pipeline.utils import init_wandb_run
from energy_consumption_forecasting.utils import get_env_var, save_json_data

logger = get_logger(name=Path(__file__).name)
ROOT_DIRPATH = Path(get_env_var(key="PROJECT_ROOT_DIR_PATH", default_value="."))

# Getting arguments for model tuning and hyperparameter tuning
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
    "--n_trials",
    type=int,
    default=20,
    help="Number of trials to study and get the best hyperparameters, "
    "needs to be in integer format.",
)

parser.add_argument(
    "-f",
    "--filepath",
    type=Path,
    default=ROOT_DIRPATH / "data" / "assets" / "hyperparameter" / "best_config.json",
    help="Filepath to save the best hyperparameter config in local directory as a "
    "JSON file, argument needs to have a .json extension.",
)

args = parser.parse_args()

logger.info("Getting the dataset from feature store.")

# Getting the training and testing dataframe
y_train, y_test, X_train, X_test = load_prepared_dataset_from_feature_store(
    feature_view_name=args.views_name,
    feature_view_ver=args.views_ver,
    training_dataset_ver=args.dataset_ver,
    target_feature=args.target_feature,
    forecasting_horizon=args.fh,
)

logger.info("Train and test dataset is available.")


@log_exception(logger=logger)
@validate_call(config=dict(arbitrary_types_allowed=True))
def model_tuning(
    X_train: pd.DataFrame = X_train,
    y_train: pd.DataFrame = y_train,
    X_test: pd.DataFrame = X_test,
    y_test: pd.DataFrame = y_test,
    fh: int = 24,
    summarize_period: List[int] = [24, 48, 72],
    model_params: Optional[Dict[str, Any]] = None,
) -> float:
    """
    This function build the lightgbm model and trains and predict on training and
    testing dataset, using the predict data then calculates an MAPE error.

    Parameters
    ----------
    X_train: pd.DataFrame, default=X_train
        A pandas dataframe for training the model not containing the target feature.

    y_train: pd.DataFrame, default=y_train
        A pandas dataframe for training the model containing the target feature.

    X_test: pd.DataFrame, default=X_test
        A pandas dataframe for testing the model not containing the target feature.

    y_test: pd.DataFrame, default=y_test
        A pandas dataframe for testing the model containing the target feature.

    fh: int, default=24
        A period that indicates the forecast horizon while making prediction in Hours.

    summarize_period: List[int], default=[24, 48, 72]
        The period at which the window summarizer transformer will be applied and
        calculate the lag, mean and std.
        For eg. [24, 48, 72] indicate: lag of 72 period, mean and std of first
        24 period, then 48 period and last 72 period.
        Note: Period can be sort of duration, in the above example it is hours.

    model_params: Optional[Dict[str, Any]], default=None
        A dict containing the hyperparameters of the LightGBM model.
        If None is provided then model is build with default parameters.

    Returns
    -------
    float
        MAPE error is returned, this indicates the error deviation between the
        actual and predicted values. Error value needs to be between 0 and 1
        and the value should be lower for better results.

    """

    # Building the model forecasting pipeline
    logger.info(
        f"Building LightGBM model with following parameters: {model_params} "
        f"and summarize period: {summarize_period}"
    )

    pipe = build_lightgbm_model(
        summarize_period=summarize_period,
        model_params=model_params,
    )

    # Training and prediction using the forecasting pipeline and
    # getting the error deviation
    fh = np.arange(fh) + 1

    pipe_return = pipe.fit(y=y_train, X=X_train, fh=fh)
    y_pred = pipe_return.predict(X=X_test)

    error = mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred, symmetric=False
    )

    logger.info(f"Model is been evaluated and the MAPE error is: {error}")

    return error


def objective(trial: optuna.trial.Trial):

    lgbm_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 500),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 1),
    }

    error = model_tuning(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        fh=args.fh,
        summarize_period=args.summarize_period,
        model_params=lgbm_params,
    )

    return error


@log_exception(logger=logger)
@validate_call
def hyperparameter_tuning(filepath: Path = args.filepath):
    """
    This function tunes the model for finding the best hyperparameters and saves the
    final best configuration locally as a JSON file and in WandB artifact.

    Parameters
    ----------
    filepath: Path, default='data/assets/hyperparameter/best_config.json'
        The filepath to save the configuration as JSON file, the path needs to have a
        .json extension at the end of the filename.
    """

    with init_wandb_run(
        run_name="get_best_hyperparameter",
        job_type="model_tuning",
        group="model",
    ) as run:

        logger.info(
            f"Details of the wandb run: run ID: {run.id}, run name: {run.name}, "
            f"project name: {run.project}, entity: {run.entity}, run group: "
            f"{run.group}, run save directory: {run.dir}, run URL: {run.url}"
        )

        # Using optuna to find the best hyperparameters and tracking it with WandB
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)

        # saving the best params locally and also logging it as an artifact
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_json_data(data=study.best_params, filepath=filepath)

        artifact = wandb.Artifact(
            name="best_config", type="model", metadata=study.best_params
        )
        artifact.add_file(local_path=filepath)
        run.log_artifact(artifact)

        logger.info(f"Best params for the LightGBM model is: {study.best_params}")
        logger.info(
            f"Hyperparameters has been saved to filepath: {filepath} and "
            "Artifact best_config has been logged successfully"
        )

        run.summary["best_mape"] = study.best_value

        run.log(
            {
                "optimization_history_plot": plot_optimization_history(study),
                "param_importance_plot": plot_param_importances(study),
                "features_slice_plot": plot_slice(study),
            }
        )

        run.finish()


if __name__ == "__main__":

    hyperparameter_tuning()
