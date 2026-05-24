import plotly.express as px
import pandas as pd
import streamlit as st

from visualization.common import get_by_dataset


def _with_relative_degradation(utility_metrics):
    rows = []

    for model_name in utility_metrics["model"].unique():
        df_model = utility_metrics[utility_metrics["model"] == model_name]
        baseline = get_by_dataset(df_model, "baseline")
        baseline_mae = baseline["mae"]

        for _, row in df_model.iterrows():
            row_dict = row.to_dict()
            row_dict["relative_mae_degradation"] = abs(row["mae"] - baseline_mae) / baseline_mae
            rows.append(row_dict)

    return pd.DataFrame(rows)


def render_utility(utility_metrics, attack_metrics, metadata):
    st.header("Utility Analysis")
    st.write("Como a utilidade dos modelos varia entre baseline e datasets privatizados?")

    utility_df = _with_relative_degradation(utility_metrics)

    st.subheader("Evolução de MAE")
    fig_mae = px.line(
        utility_df,
        x="dataset",
        y="mae",
        color="model",
        markers=True,
        title="MAE por dataset",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    st.subheader("Evolução de RMSE")
    fig_rmse = px.line(
        utility_df,
        x="dataset",
        y="rmse",
        color="model",
        markers=True,
        title="RMSE por dataset",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Degradação relativa de MAE")
    fig_degradation = px.bar(
        utility_df[utility_df["dataset"] != "baseline"],
        x="dataset",
        y="relative_mae_degradation",
        color="model",
        barmode="group",
        title="Degradação relativa em relação ao baseline",
    )
    st.plotly_chart(fig_degradation, use_container_width=True)

    st.subheader("Heatmap de MAE")
    heatmap_df = utility_df.pivot(index="model", columns="dataset", values="mae")
    fig_heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="MAE por modelo e dataset",
        labels={"color": "MAE"},
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
