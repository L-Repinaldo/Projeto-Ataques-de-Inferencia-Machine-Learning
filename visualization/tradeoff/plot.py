import plotly.express as px

from visualization.common import build_tradeoff_points


def build_tradeoff_figure(utility_results, attack_results):
    """
    Trade-off direto Utilidade X Vazamento:
    X = perda relativa de utilidade (MAE)
    Y = risco de MIA (advantage)
    Todos os modelos no mesmo gráfico.
    """
    tradeoff_df = build_tradeoff_points(
        utility_results=utility_results,
        attack_results=attack_results,
    )

    fig = px.line(
        tradeoff_df,
        x="utility_loss",
        y="advantage",
        color="model",
        markers=True,
        text=tradeoff_df["epsilon"].map(lambda value: f"ε={value}"),
        hover_data={
            "model": True,
            "dataset": True,
            "epsilon": True,
            "utility_loss": ":.4f",
            "advantage": ":.4f",
            "mae": ":.4f",
            "rmse": ":.4f",
            "attack_acc": ":.4f",
        },
        labels={
            "utility_loss": "Perda relativa de utilidade (MAE)",
            "advantage": "Risco de vazamento (Attack Accuracy)",
            "model": "Modelo",
        },
        title="Trade-off Vazamento vs Utilidade",
    )

    fig.update_traces(textposition="top left")
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Aleatório")
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="Ruído empírico")
    fig.add_hline(y=0.15, line_dash="dot", line_color="orange", annotation_text="Vazamento moderado")
    fig.add_hline(y=0.4, line_dash="dot", line_color="red", annotation_text="Vazamento alto")
    fig.update_layout(
        legend_title_text="Modelo",
        hovermode="closest",
        template="plotly_white",
    )
    return fig


def plot_privacy_utility_tradeoff(utility_results, attack_results):
    fig = build_tradeoff_figure(
        utility_results=utility_results,
        attack_results=attack_results,
    )
    fig.show()
    return fig
