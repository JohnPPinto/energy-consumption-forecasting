import pandas as pd
import plotly.express as px
import requests
from yarl import URL

from utils import build_dataframe


def build_monitoring_plot(api_url: URL, municipality_num: int, branch: int):
    """
    This function builds the monitoring plot component of prediction data.

    Parameters
    ----------
    api_url: yarl.URL
        The API URL to access the FastAPI server.

    municipality_num: int
        The municipality number to access the data and send request to server.

    branch: int
        The branch number to access the data and send request to server.

    Returns
    -------
    plotly.graph_objects.Figure
        A plotly line figure containing the representation for ground truth and
        prediction data.
    """

    # Getting the prediction from API by providing the argument values
    response = requests.get(
        url=api_url / "monitor" / "prediction" / str(municipality_num) / str(branch),
        verify=False,
    )

    if response.status_code != 200:
        # Building a empty dataframe for no data found status
        ground_truth_df = build_dataframe({"datetime": [], "consumption": []})
        cached_prediction_df = build_dataframe({"datetime": [], "consumption": []})
        title = "No Data Available for the provided Municipality Number and Branch"

    else:
        json_response = response.json()

        # Building the dataframe from the received data from the API
        ground_truth_datetime = json_response.get("ground_truth_datetime")
        ground_truth_consumption = json_response.get("ground_truth_consumption")
        cached_prediction_datetime = json_response.get("cached_prediction_datetime")
        cached_prediction_consumption = json_response.get(
            "cached_prediction_consumption"
        )

        ground_truth_df = build_dataframe(
            {
                "datetime": ground_truth_datetime,
                "consumption": ground_truth_consumption,
            },
        )
        cached_prediction_df = build_dataframe(
            {
                "datetime": cached_prediction_datetime,
                "consumption": cached_prediction_consumption,
            },
        )
        title = "Monitoring - Denmark's Energy Consumption Forecasting Performance"

    dataframe_dict = {
        "Ground Truth": ground_truth_df,
        "Prediction": cached_prediction_df,
    }
    dataframe: iter = (df.assign(data=k) for k, df in dataframe_dict.items())
    dataframe = pd.concat(dataframe)

    # Building the plot using the available dataframe
    fig = px.line(
        data_frame=dataframe,
        x="datetime",
        y="consumption",
        color="data",
        color_discrete_sequence=["deepskyblue", "springgreen"],
        title=title,
    )

    fig.update_traces(
        line={"width": 4},
        hovertemplate="Datetime: %{x} <br>Energy Consumption: %{y} kWh",
    )
    fig.update_xaxes(
        title_text="Datetime-Denmark",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(step="all"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                ]
            )
        ),
    )
    fig.update_yaxes(title_text="Energy Consumption (kWh)")
    fig.update_layout(legend_title=None)

    return fig


def build_metrics_plot(api_url: URL):
    """
    This function builds the monitoring plot component of metrics data.

    Parameters
    ----------
    api_url: yarl.URL
        The API URL to access the FastAPI server.

    Returns
    -------
    plotly.graph_objects.Figure
        A plotly line figure containing the representation of MAPE and RMSPE metrics data.
    """

    # Getting the prediction from API by providing the argument values
    response = requests.get(url=api_url / "monitor" / "metrics", verify=False)

    if response.status_code != 200:
        # Building a empty dataframe for no data found status
        metrics_df = build_dataframe({"datetime": [], "mape": [], "rmspe": []})
        title = "No Data Available for the provided Municipality Number and Branch"

    else:
        json_response = response.json()

        # Building the dataframe from the received data from the API
        datetime = json_response.get("datetime")
        mape = json_response.get("mape")
        rmspe = json_response.get("rmspe")

        metrics_df = build_dataframe(
            {
                "datetime": datetime,
                "mape": [i * 100 for i in mape],
                "rmspe": [i * 100 for i in rmspe],
            }
        )
        title = "Monitoring Metrics(MAPE and RMSPE) - Denmark's Energy Consumption Forecasting Performance"

    # Building the plot using the available dataframe
    fig = px.bar(
        data_frame=metrics_df,
        x="datetime",
        y=["mape", "rmspe"],
        opacity=0.9,
        orientation="v",
        barmode="group",
        color_discrete_sequence=["deepskyblue", "springgreen"],
        title=title,
    )

    fig.update_traces(
        hovertemplate="Datetime: %{x} <br>Value: %{y:.2f} %",
    )
    fig.update_xaxes(
        title_text="Datetime-Denmark",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(step="all"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                ]
            )
        ),
    )
    fig.update_yaxes(title_text="Metrics (%)")
    fig.update_layout(legend_title=None)

    return fig
