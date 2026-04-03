from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Neural Traffic Prediction System", layout="wide")

st.title("Neural Traffic Prediction System")
st.subheader("London Traffic Forecasting and Hotspot Analysis")

output_path = Path("outputs")

comparison_df = pd.read_csv(output_path / "final_model_comparison_with_lstm.csv")
hotspot_summary = pd.read_csv(output_path / "hotspot_summary.csv")
hotspot_points = pd.read_csv(output_path / "hotspot_points.csv")

st.write("## Model Comparison")
st.dataframe(comparison_df, use_container_width=False)

st.write("## Hotspot Summary")
st.dataframe(hotspot_summary, use_container_width=False)

col1, col2 = st.columns(2)

with col1:
    error_df = comparison_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Value"
    )

    fig_error = px.bar(
        error_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Error Comparison",
        width=500,
        height=350
    )
    st.plotly_chart(fig_error, use_container_width=True)

with col2:
    fig_r2 = px.bar(
        comparison_df,
        x="Model",
        y="R2",
        title="Model R² Comparison",
        width=500,
        height=350
    )
    st.plotly_chart(fig_r2, use_container_width=True)

st.write("## Hotspot Clusters")

fig_hotspot = px.scatter(
    hotspot_points,
    x="longitude",
    y="latitude",
    color=hotspot_points["cluster"].astype(str),
    hover_data=["count_point_id", "all_motor_vehicles"],
    title="Traffic Hotspot Clusters in London",
    width=800,
    height=500
)

st.plotly_chart(fig_hotspot, use_container_width=True)