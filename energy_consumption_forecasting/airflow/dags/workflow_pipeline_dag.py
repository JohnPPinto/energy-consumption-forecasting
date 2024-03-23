from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable


@dag(
    dag_id="workflow_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=True,
    tags=["feature-pipeline", "training-pipeline", "inference-pipeline"],
    max_active_runs=1,
    default_args={
        "retries": 60,
        "retry_delay": timedelta(days=1),
    },
)
def workflow_pipeline():
    """
    This function creates a DAG named workflow_pipeline within Apache Airflow.
    It is schedule to run daily and the start date is 2024-01-01.

    This workflow pipeline dag builds the lifecycle of the project by running the
    feature pipeline, training pipeline and inference pipeline sequentially.
    """

    # Creating a task for running the feature pipeline
    @task.virtualenv(
        task_id="run_feature_pipeline",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_feature_pipeline(
        start_date_time: str,
        end_date_time: str,
        feature_group_name: str,
        feature_group_ver: int,
    ) -> dict:
        """
        This function calls the feature pipeline module and performs the ETL process.
        """

        from pathlib import Path

        from energy_consumption_forecasting.airflow.dags.utils import transform_datetime
        from energy_consumption_forecasting.feature_pipeline import feature_pipeline
        from energy_consumption_forecasting.logger import get_logger

        logger = get_logger(name=Path(__file__).name)

        start_date_time = transform_datetime(date_time=str(start_date_time))
        end_date_time = transform_datetime(date_time=str(end_date_time))

        logger.info(
            "Running feature pipeline in apache airflow with following arguments:"
        )
        logger.info(f"start_date_time = {start_date_time}")
        logger.info(f"end_date_time = {end_date_time}")
        logger.info(f"feature_group_name = {feature_group_name}")
        logger.info(f"feature_group_ver = {feature_group_ver}")

        metadata, filepath = feature_pipeline.run_feature_pipeline(
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            feature_group_name=feature_group_name,
            feature_group_ver=feature_group_ver,
        )

        logger.info(
            "Feature pipeline is completed and metadata is saved as a "
            f'Json file: "{filepath}"'
        )

        return metadata

    # Creating a task for building a feature view and training dataset
    @task.virtualenv(
        task_id="create_feature_view",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def create_feature_view(
        start_datetime: str,
        feature_views_name: str,
        feature_pipeline_metadata: dict,
    ) -> dict:
        """
        This function calls the feature view module and creates a training dataset.
        """

        from pathlib import Path

        from energy_consumption_forecasting.airflow.dags.utils import transform_datetime
        from energy_consumption_forecasting.feature_pipeline import feature_view
        from energy_consumption_forecasting.logger import get_logger

        logger = get_logger(name=Path(__file__).name)

        start_datetime = transform_datetime(date_time=start_datetime)
        end_datetime = transform_datetime(
            date_time=feature_pipeline_metadata.get("data_extraction_end_datetime")
        )

        logger.info(
            "Creating feature view and training dataset in apache airflow with "
            "following arguments:"
        )
        logger.info(f"start_date_time = {start_datetime}")
        logger.info(f"end_date_time = {end_datetime}")
        logger.info(f"feature_views_name = {feature_views_name}")
        logger.info(
            f"feature_group_version = {feature_pipeline_metadata.get('version')}"
        )
        logger.info(f"feature_group_name = {feature_pipeline_metadata.get('name')}")
        logger.info(
            f"feature_views_description = {feature_pipeline_metadata.get('description')}"
        )

        metadata, filepath = feature_view.create_feature_view(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            feature_group_version=feature_pipeline_metadata.get("version"),
            feature_group_name=feature_pipeline_metadata.get("name"),
            feature_views_name=feature_views_name,
            feature_views_description=feature_pipeline_metadata.get("description"),
        )

        logger.info(
            "Feature view is build and metadata is saved as a "
            f'Json file: "{filepath}"'
        )

        return metadata

    # Creating a task for performing hyperparameter tuning
    @task.virtualenv(
        task_id="run_hyperparameter_tuning",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_hyperparameter_tuning(
        target_feature: str,
        forecasting_horizon: int,
        summarize_period: list,
        n_trials: int,
        feature_view_metadata: dict,
    ) -> dict:
        """
        This function calls the hyperparameter_tuning module and performs a series of
        experiment or trials to get the best model parameters for LightGBM model.
        """

        from pathlib import Path

        from energy_consumption_forecasting.logger import get_logger
        from energy_consumption_forecasting.training_pipeline import (
            hyperparameter_tuning,
        )

        logger = get_logger(name=Path(__file__).name)

        logger.info(
            "Running hyperparameter tuning in apache airflow with "
            "following arguments:"
        )
        logger.info(f"feature_view_name = {feature_view_metadata.get('name')}")
        logger.info(f"feature_view_ver = {feature_view_metadata.get('version')}")
        logger.info(
            f"training_dataset_ver = {feature_view_metadata.get('train_dataset_version')}"
        )
        logger.info(f"target_feature = {target_feature}")
        logger.info(f"forecasting_horizon = {forecasting_horizon}")
        logger.info(f"summarize_period = {summarize_period}")
        logger.info(f"n_trials = {n_trials}")

        model_params, filepath = hyperparameter_tuning.run_hyperparameter_tuning(
            feature_view_name=feature_view_metadata.get("name"),
            feature_view_ver=feature_view_metadata.get("version"),
            training_dataset_ver=feature_view_metadata.get("train_dataset_version"),
            target_feature=target_feature,
            forecasting_horizon=forecasting_horizon,
            summarize_period=summarize_period,
            n_trials=n_trials,
        )

        logger.info(
            "Hyperparameter tuning is been completed and the parameters are saved "
            f'in a Json file: "{filepath}"'
        )

        return model_params

    # Creating a task for performing model training
    @task.virtualenv(
        task_id="run_training_pipeline",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_training_pipeline(
        fh: int,
        summarize_period: list,
        target_feature: str,
        model_name: str,
        model_id_or_ver: int,
        model_params_or_path: dict,
        feature_view_metadata: dict,
    ):
        """
        This function calls the training_pipeline module and performs the model training
        on the LightGBM model.
        """

        from pathlib import Path

        from energy_consumption_forecasting.logger import get_logger
        from energy_consumption_forecasting.training_pipeline import training_pipeline

        logger = get_logger(name=Path(__file__).name)

        logger.info(
            "Running training pipeline in apache airflow with " "following arguments:"
        )
        logger.info(f"fh = {fh}")
        logger.info(f"summarize_period = {summarize_period}")
        logger.info(f"feature_view_name = {feature_view_metadata.get('name')}")
        logger.info(f"feature_view_ver = {feature_view_metadata.get('version')}")
        logger.info(
            f"training_dataset_ver = {feature_view_metadata.get('train_dataset_version')}"
        )
        logger.info(f"target_feature = {target_feature}")
        logger.info(f"model_name = {model_name}")
        logger.info(f"model_id_or_ver = {model_id_or_ver}")
        logger.info(f"model_params_path = {model_params_or_path}")

        metadata, filepath = training_pipeline.model_training_pipeline(
            fh=fh,
            summarize_period=summarize_period,
            feature_view_name=feature_view_metadata.get("name"),
            feature_view_ver=feature_view_metadata.get("version"),
            training_dataset_ver=feature_view_metadata.get("train_dataset_version"),
            target_feature=target_feature,
            model_name=model_name,
            model_id_or_ver=model_id_or_ver,
            model_params_or_path=model_params_or_path,
        )

        logger.info(
            "Training pipeline is been completed and the metadata has been saved "
            f'in a Json file: "{filepath}"'
        )

        return metadata

    # Creating a task for performing inference on the model
    @task.virtualenv(
        task_id="run_inference_pipeline",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        system_site_packages=False,
    )
    def run_inference_pipeline(training_pipeline_metadata: dict):
        """
        This function calls the inference_pipeline module and performs inference on the
        trained model using the testing data that was selected durning training.
        """

        from pathlib import Path

        from energy_consumption_forecasting.airflow.dags.utils import transform_datetime
        from energy_consumption_forecasting.inference_pipeline import inference_pipeline
        from energy_consumption_forecasting.logger import get_logger

        logger = get_logger(name=Path(__file__).name)

        # Getting the experiment metadata
        exp_metadata = training_pipeline_metadata.get("experiment")

        # Transforming the start and end datetime
        start_datetime = transform_datetime(
            date_time=exp_metadata.get("training_start_datetime")
        )
        end_datetime = transform_datetime(
            date_time=exp_metadata.get("testing_end_datetime")
        )

        logger.info(
            "Running inference pipeline in apache airflow with following arguments:"
        )
        logger.info(f"start_datetime = {start_datetime}")
        logger.info(f"end_datetime = {end_datetime}")
        logger.info(f"fh = {exp_metadata.get('fh')}")
        logger.info(f"target_feature = {exp_metadata.get('target_feature')}")
        logger.info(f"feature_view_name = {exp_metadata.get('feature_view_name')}")
        logger.info(f"feature_view_ver = {exp_metadata.get('feature_view_ver')}")
        logger.info(f"model_name = {exp_metadata.get('model_name')}")
        logger.info(f"model_version = {exp_metadata.get('model_id_or_ver')}")

        inference_pipeline.run_inference_pipeline(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            fh=exp_metadata.get("fh"),
            target_feature=exp_metadata.get("target_feature"),
            feature_view_ver=exp_metadata.get("feature_view_ver"),
            feature_view_name=exp_metadata.get("feature_view_name"),
            model_version=exp_metadata.get("model_id_or_ver"),
            model_name=exp_metadata.get("model_name"),
        )

        logger.info("Inference pipeline task has been completed.")

    # Creating a task for performing inference performance monitoring
    @task.virtualenv(
        task_id="run_monitor_performance",
        requirements=[
            "--trusted-host 172.17.0.1:8000",
            "--index-url http://${PYPI_USER}:${PYPI_PASSWORD}@172.17.0.1:8000",
            "energy_consumption_forecasting",
        ],
        python_version="3.11",
        system_site_packages=False,
    )
    def run_monitor_performance(training_pipeline_metadata: dict):
        """
        This function calls the monitor_performance module and computes the MAPE and
        RMSPE error on the cached prediction and ground truth data.
        """

        from pathlib import Path

        from energy_consumption_forecasting.airflow.dags.utils import transform_datetime
        from energy_consumption_forecasting.inference_pipeline import (
            monitor_performance,
        )
        from energy_consumption_forecasting.logger import get_logger

        logger = get_logger(name=Path(__file__).name)

        # Getting the experiment metadata
        exp_metadata = training_pipeline_metadata.get("experiment")

        logger.info(
            "Running monitor performance in apache airflow with following arguments:"
        )
        logger.info(f"target_feature = {exp_metadata.get('target_feature')}")
        logger.info(f"feature_view_name = {exp_metadata.get('feature_view_name')}")
        logger.info(f"feature_view_ver = {exp_metadata.get('feature_view_ver')}")

        monitor_performance.compute_performance_metrics(
            target_feature=exp_metadata.get("target_feature"),
            feature_view_ver=exp_metadata.get("feature_view_ver"),
            feature_view_name=exp_metadata.get("feature_view_name"),
        )

        logger.info("Monitor performance task has been completed.")

    # Getting airflow variables for passing it as arguments in the task
    feature_group_name = str(
        Variable.get(
            key="workflow_pipeline_feature_group_name",
            default_var="denmark_energy_consumption_group",
        )
    )

    feature_group_ver = int(
        Variable.get(
            key="workflow_pipeline_feature_group_ver",
            default_var=1,
        )
    )

    feature_views_name = str(
        Variable.get(
            key="workflow_pipeline_feature_views_name",
            default_var="denmark_energy_consumption_view",
        )
    )

    train_dataset_start_date = str(
        Variable.get(
            key="workflow_pipeline_train_dataset_start_date",
            default_var="2023-12-01 00:00:00",
        )
    )

    target_feature = str(
        Variable.get(
            key="workflow_pipeline_target_feature",
            default_var="consumption_kwh",
        )
    )

    forecasting_horizon = int(
        Variable.get(
            key="workflow_pipeline_forecasting_horizon",
            default_var=24,
        )
    )

    summarize_period = dict(
        Variable.get(
            key="workflow_pipeline_summarize_period",
            default_var={"0": [24, 48, 72]},
            deserialize_json=True,
        )
    )
    summarize_period = list(summarize_period.get("0"))

    n_trials = int(
        Variable.get(
            key="workflow_pipeline_n_trials",
            default_var=20,
        )
    )

    model_name = str(
        Variable.get(
            key="workflow_pipeline_model_name",
            default_var="lightgbm_model",
        )
    )

    model_ver = int(
        Variable.get(
            key="workflow_pipeline_model_ver",
            default_var=1,
        )
    )

    # Running feature pipeline task
    feature_pipeline_metadata = run_feature_pipeline(
        start_date_time="{{ dag_run.logical_date }}",
        end_date_time="{{ dag_run.logical_date }}",
        feature_group_name=feature_group_name,
        feature_group_ver=feature_group_ver,
    )

    # Running feature view and training dataset task
    feature_views_metadata = create_feature_view(
        start_datetime=train_dataset_start_date,
        feature_views_name=feature_views_name,
        feature_pipeline_metadata=feature_pipeline_metadata,
    )

    # Running Hyperparameter tuning task
    lightgbm_params = run_hyperparameter_tuning(
        target_feature=target_feature,
        forecasting_horizon=forecasting_horizon,
        summarize_period=summarize_period,
        n_trials=n_trials,
        feature_view_metadata=feature_views_metadata,
    )

    # Running the training pipeline task
    model_train_metadata = run_training_pipeline(
        fh=forecasting_horizon,
        summarize_period=summarize_period,
        target_feature=target_feature,
        model_name=model_name,
        model_id_or_ver=model_ver,
        model_params_or_path=lightgbm_params,
        feature_view_metadata=feature_views_metadata,
    )

    # Running the inference pipeline task
    inference_pipeline = run_inference_pipeline(
        training_pipeline_metadata=model_train_metadata
    )

    # Running the monitor performance task
    monitor_performance = run_monitor_performance(
        training_pipeline_metadata=model_train_metadata
    )

    # Defining the DAG dependencies structure
    (
        feature_pipeline_metadata
        >> feature_views_metadata
        >> lightgbm_params
        >> model_train_metadata
        >> inference_pipeline
        >> monitor_performance
    )


workflow_pipeline()
