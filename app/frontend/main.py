import requests
import streamlit as st
from frontend.components import build_prediction_plot
from yarl import URL

API_URL = URL(val="http://172.17.0.1:8001/api/v1")
TITLE = "Denmark's Energy Consumption Forecasting"

st.set_page_config(
    page_title=TITLE,
    page_icon="📈",
    layout="wide",
)
st.title(body=TITLE)

# Getting the municipality number and branch list
municipality_num_response = requests.get(url=API_URL / "municipality_number_values")
branch_response = requests.get(url=API_URL / "branch_values")

# Creating a drop down for municipality number selection
municipality_num = st.selectbox(
    label="**Municipality Number:** Each of the 98 Danish municipalities have a "
    "unique number, ranging from 101 Copenhagen to 860 Hj\u00f8rring.",
    options=municipality_num_response.json().get("values", []),
    index=None,
    placeholder="Select a municipality number...",
)

# Creating a drop down for branch selection
branch_maps = {
    1: "Offentligt - Public",
    2: "Erhverv - Industry",
    3: "Privat - Private",
}
branch_list = [branch_maps.get(i) for i in branch_response.json().get("values", [])]
branch = st.selectbox(
    label="**Branch:** All measurement id's related to a Central Business Register(CVR) "
    "number are labeled as 'Erhverv' except those that belong to 'Office, admin, "
    "which are labeled 'Offentligt'. All other measurement id's are labeled 'Privat'.",
    options=branch_list,
    index=None,
    placeholder="Select a branch...",
)

# Getting the plots by providing the selected values of municipality num and branch
if municipality_num and branch:
    branch = list(filter(lambda x: branch_maps[x] == branch, branch_maps))[0]

    with st.status("Generating your chart, wait for it...") as status:
        st.plotly_chart(
            figure_or_data=build_prediction_plot(
                api_url=API_URL,
                municipality_num=municipality_num,
                branch=branch,
            ),
            use_container_width=True,
        )
        status.update(label="Your chart is ready!", state="complete", expanded=True)
