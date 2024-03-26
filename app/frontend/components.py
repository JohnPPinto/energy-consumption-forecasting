import pandas as pd
import plotly.express as px
import requests
from yarl import URL

from utils import build_dataframe


def build_prediction_plot(api_url: URL, municipality_num: int, branch: int):
    """
    This function builds the prediction plot component.

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
        A plotly line figure containing the representation for historical and
        prediction data.
    """

    # Getting the prediction from API by providing the argument values
    response = requests.get(
        url=api_url / "prediction" / str(municipality_num) / str(branch), verify=False
    )

    if response.status_code != 200:
        # Building a empty dataframe for no data found status
        historical_df = build_dataframe({"datetime": [], "consumption": []})
        prediction_df = build_dataframe({"datetime": [], "consumption": []})
        title = "No Data Available for the provided Municipality Number and Branch"

    else:
        json_response = response.json()

        # Building the dataframe from the received data from the API
        historical_datetime = json_response.get("historical_datetime")
        historical_consumption = json_response.get("historical_consumption")
        prediction_datetime = json_response.get("prediction_datetime")
        prediction_consumption = json_response.get("prediction_consumption")

        historical_df = build_dataframe(
            {
                "datetime": historical_datetime,
                "consumption": historical_consumption,
            }
        )
        prediction_df = build_dataframe(
            {
                "datetime": prediction_datetime,
                "consumption": prediction_consumption,
            }
        )
        title = "Denmark's Energy Consumption Forecasting for Next 24 Hours"

    dataframe_dict = {"historical": historical_df, "prediction": prediction_df}
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
