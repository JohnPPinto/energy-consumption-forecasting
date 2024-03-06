from typing import Any, Dict, List, Optional

import lightgbm as lgb
from sktime.forecasting.compose import ForecastingPipeline, make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer


def build_naive_forecast_model(seasonal_periodicity: int = 1) -> NaiveForecaster:
    """
    This function builds a naive forecast model using the 'last' as a strategy,
    where the forecast will be based on the last value of each season in the series.

    Parameters
    ----------
    seasonal_periodicity: int, default=1
        A integer for defining the seasonal period of the series forecasting.

    Returns
    -------
    sktime.forecasting.naive.NaiveForecaster
        Returns the NaiveForecaster model with the specified seasonal period.
    """
    return NaiveForecaster(sp=seasonal_periodicity)


def build_lightgbm_model(
    n_jobs: int = -1,
    summarize_period: List[int] = [24, 48, 72],
    model_params: Optional[Dict[str, Any]] = None,
):
    """
    This function builds a LightGBM model using the sktime method and returns a
    forecasting pipeline which can be used to fit and predict the time series data.

    Parameters
    ----------
    n_jobs: int, default=-1
        The number of jobs to run in parallel for applying the window functions.
        -1 means using all processors.

    lag_period: List[int], default=[24, 48, 72]
        The period at which the window summarizer transformer will be applied and
        calculate the lag, mean and std.
        For eg. [24, 48, 72] indicate: lag of 72 period, mean and std of first
        24 period, then 48 period and last 72 period.
        Note: Period can be sort of duration, in the above example it is hours.

    model_params: Dict[str, Any] or None, default=None
        A dict containing the hyperparameters of the LightGBM model.
        If None is provided then model is build with default parameters.

    Returns
    -------
    sktime.forecasting.compose.ForecastingPipeline
        A sktime forecasting pipeline is returned containing the LightGBM model and
        all the necessary transformer, this can be used for training and predicting.
    """

    # Creating window summarizer to transform the dataframe
    kwargs = {
        "lag_feature": {
            "lag": list(range(1, summarize_period[-1] + 1)),
            "mean": [[1, i] for i in summarize_period],
            "std": [[1, i] for i in summarize_period],
        }
    }

    window_summarizer = WindowSummarizer(**kwargs, n_jobs=n_jobs)

    # Building the LightGBM model on sktime forecaster method
    if model_params is None:
        lgbm_regressor = lgb.LGBMRegressor()
    else:
        lgbm_regressor = lgb.LGBMRegressor(**model_params)

    forecaster = make_reduction(
        estimator=lgbm_regressor,
        transformers=[window_summarizer],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    # Building the forecast pipeline for fitting the data in the model
    pipe = ForecastingPipeline(
        steps=[
            (
                "daily_season",
                DateTimeFeatures(
                    manual_selection=["day_of_week", "hour_of_day"],
                    keep_original_columns=True,
                ),
            ),
            ("forecaster", forecaster),
        ]
    )

    return pipe
