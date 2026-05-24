import pandas as pd
import streamlit as st


def render_overview(utility_metrics, attack_metrics, metadata):
    st.header("Overview")
    st.write("Resumo do artifact experimental selecionado.")

    col_dataset, col_models, col_datasets = st.columns(3)
    col_dataset.metric("Dataset version", metadata.get("dataset_version", "-"))
    col_models.metric("Modelos", len(metadata.get("modelos", [])))
    col_datasets.metric("Datasets avaliados", utility_metrics["dataset"].nunique())

    st.subheader("Protocolo Experimental")
    protocol = {
        "Timestamp": metadata.get("timestamp", "-"),
        "Seeds": ", ".join(map(str, metadata.get("seeds", []))),
        "Test sizes": ", ".join(map(str, metadata.get("test_sizes", []))),
        "Modelos": ", ".join(metadata.get("modelos", [])),
        "Datasets": ", ".join(utility_metrics["dataset"].unique()),
    }
    st.table(pd.DataFrame(protocol.items(), columns=["Campo", "Valor"]))

    st.subheader("Métricas Persistidas")
    col_utility, col_attack = st.columns(2)
    with col_utility:
        st.caption("Utility metrics")
        st.dataframe(utility_metrics, use_container_width=True)
    with col_attack:
        st.caption("Attack metrics")
        st.dataframe(attack_metrics, use_container_width=True)
