import streamlit as st

from visualization.plots.tradeoff import build_tradeoff_figure


def render_tradeoff(utility_metrics, attack_metrics, metadata):
    st.header("Trade-off Analysis")
    st.write("Qual é a relação entre perda relativa de utilidade e risco de vazamento?")

    fig = build_tradeoff_figure(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )
    st.plotly_chart(fig, use_container_width=True)
