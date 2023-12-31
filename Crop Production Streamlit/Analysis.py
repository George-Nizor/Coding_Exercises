import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import numpy as np
import json
from streamlit_lottie import st_lottie

# run with python -m streamlit run Analysis.py


# NOTES FOR FURTHER DEVELOPMENT
# add regression code for vegetable and w heat production, write a function that performs the regression analysis
# and write another that plots the chart, have the passed variables be the axis and call the functions
# for each of the types


# _______________________________________________Analysis Code____________________________________________________________

## Data Load
df = pd.read_csv("world_food_production.csv")


def generate_regression(x, y, x2):
    model = LinearRegression()
    model.fit(x, y)
    predicted_x = model.predict(x2)
    return predicted_x


def generate_regression_plot(x_, y_, y2, x2, titlex="", titley="", title_=""):
    ## Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_, y=y2, mode="markers", name="Actual Data"))
    fig.add_trace(
        go.Scatter(
            x=x2.flatten(),
            y=y_,
            mode="lines",
            name="Regression Line",
        )
    )
    fig.update_layout(
        title=f"{title_}",
        xaxis_title=f"{titlex}",
        yaxis_title=f"{titley}",
        showlegend=True,
    )
    return fig


## Data Processing
X = df[["year"]]
last_year = df["year"].max()
future_years = np.arange(last_year + 1, last_year + 7).reshape(-1, 1)
all_years = np.concatenate((X, future_years))


# _______________________________________________Streamlit Page Code____________________________________________________________
def load_lottie_animation(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


col_1, col_2 = st.columns([1, 1])
with col_1:
    st.markdown("""# 🌽\n # Food \n # Production   \n# Analysis\n # 🌾🚜""")
with col_2:
    lottie_animation = load_lottie_animation("working-chart.json")
    st_lottie(lottie_animation)
st.divider()
st.markdown(
    """
### 🌎World Food Production: Data table
The following data table contains global Rice, Wheat and Vegetable food production in **Million Tonnes**
"""
)
col1, col2 = st.columns([4, 1])
with col1:
    st.dataframe(df)
with col2:
    st.markdown(
        """
From the raw data it is clear that overall food production across all three groups of food has drastically increased, with vegetables almost doubling in production.
"""
    )
st.divider()
st.markdown(
    """
### 📈World Food Production: Chart
"""
)
st.line_chart(
    df,
    x="year",
    y=["rice_production", "wheat_production", "vegetable_production"],
)


predicted_rice_production = generate_regression(
    df[["year"]], df["rice_production"], all_years
)
predicted_wheat_production = generate_regression(
    df[["year"]], df["wheat_production"], all_years
)
predicted_vegetable_production = generate_regression(
    df[["year"]], df["vegetable_production"], all_years
)
st.plotly_chart(
    generate_regression_plot(
        df["year"],
        predicted_rice_production,
        df["rice_production"],
        all_years,
        "Year",
        "Rice Production",
        "Rice Production Prediction",
    )
)

st.plotly_chart(
    generate_regression_plot(
        df["year"],
        predicted_wheat_production,
        df["wheat_production"],
        all_years,
        "Year",
        "Wheat Production",
        "Wheat Production Prediction",
    )
)
st.plotly_chart(
    generate_regression_plot(
        df["year"],
        predicted_vegetable_production,
        df["vegetable_production"],
        all_years,
        "Year",
        "Vegetable Production",
        "Vegetable Production Prediction",
    )
)
