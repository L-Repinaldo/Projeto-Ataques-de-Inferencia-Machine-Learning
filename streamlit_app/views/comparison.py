import plotly.express as px
import streamlit as st

from visualization.common import build_tradeoff_points


def render_comparison(utility_metrics, attack_metrics, metadata):
    st.header("Comparison")
    st.write("Comparação cruzada entre modelos e níveis de privacidade.")

    tradeoff_points = build_tradeoff_points(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )

    st.subheader("Tabela consolidada de trade-off")
    st.dataframe(tradeoff_points, use_container_width=True)

    st.subheader("Heatmap: perda relativa de utilidade")
    utility_heatmap = tradeoff_points.pivot(index="model", columns="dataset", values="utility_loss")
    fig_utility = px.imshow(
        utility_heatmap,
        text_auto=True,
        aspect="auto",
        title="Perda relativa de utilidade por modelo e epsilon",
        labels={"color": "Perda relativa"},
    )
    st.plotly_chart(fig_utility, use_container_width=True)

    st.subheader("Heatmap: advantage")
    leakage_heatmap = tradeoff_points.pivot(index="model", columns="dataset", values="advantage")
    fig_leakage = px.imshow(
        leakage_heatmap,
        text_auto=True,
        aspect="auto",
        title="Advantage por modelo e epsilon",
        labels={"color": "Advantage"},
    )
    st.plotly_chart(fig_leakage, use_container_width=True)

    st.subheader("Dispersão comparativa")
    fig_scatter = px.scatter(
        tradeoff_points,
        x="utility_loss",
        y="advantage",
        color="model",
        symbol="dataset",
        hover_data=["epsilon", "mae", "rmse", "attack_acc"],
        title="Modelos × epsilon no espaço utilidade-vazamento",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
