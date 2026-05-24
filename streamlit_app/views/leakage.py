import plotly.express as px
import streamlit as st


def render_leakage(utility_metrics, attack_metrics, metadata):
    st.header("Leakage Analysis")
    st.write("Como o risco de inferência varia por modelo e nível de privacidade?")

    st.subheader("Attack Accuracy")
    fig_attack_acc = px.line(
        attack_metrics,
        x="dataset",
        y="attack_acc",
        color="model",
        markers=True,
        title="Acurácia do ataque por dataset",
    )
    st.plotly_chart(fig_attack_acc, use_container_width=True)

    st.subheader("Advantage")
    fig_advantage = px.line(
        attack_metrics,
        x="dataset",
        y="advantage",
        color="model",
        markers=True,
        title="Advantage por dataset",
    )
    fig_advantage.add_hline(y=0.0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_advantage, use_container_width=True)

    st.subheader("Member vs Non-member Accuracy")
    melted = attack_metrics.melt(
        id_vars=["model", "dataset"],
        value_vars=["member_acc", "non_member_acc"],
        var_name="metric",
        value_name="value",
    )
    fig_member = px.line(
        melted,
        x="dataset",
        y="value",
        color="model",
        line_dash="metric",
        markers=True,
        title="Comparação member_acc e non_member_acc",
    )
    st.plotly_chart(fig_member, use_container_width=True)

    st.subheader("Heatmap de Advantage")
    heatmap_df = attack_metrics.pivot(index="model", columns="dataset", values="advantage")
    fig_heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="Advantage por modelo e dataset",
        labels={"color": "Advantage"},
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
