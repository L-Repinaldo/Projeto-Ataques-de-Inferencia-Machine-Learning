import plotly.express as px

from visualization.common import build_tradeoff_points


def plot_privacy_utility_tradeoff(utility_results, attack_results):
    """
    Trade-off direto Utilidade X Vazamento.

    X = perda relativa de utilidade (test_MAE)
    Y = risco de vazamento (advantage)
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
        hover_data={
            "epsilon": True,
            "test_mae": ":.0f",
            "attack_acc": ":.3f",
            "utility_loss": ":.3f",
            "advantage": ":.3f",
        },
        labels={
            "utility_loss": "Perda relativa de utilidade",
            "advantage": "Advantage",
            "model": "Modelo",
        },
        title="Privacy–Utility Trade-off AFA",
    )

    # Faixas de referência para interpretação
    fig.add_hline(
        y=0.00,
        line_dash="dash",
        line_color="gray",
    )

    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="green",
    )

    fig.add_hline(
        y=0.15,
        line_dash="dot",
        line_color="orange",
    )

    fig.add_hline(
        y=0.40,
        line_dash="dot",
        line_color="red",
    )

    fig.add_annotation(
        xref="paper",
        x=1.01,
        y=0.00,
        text="Aleatório",
        showarrow=False,
    )

    fig.add_annotation(
        xref="paper",
        x=1.01,
        y=0.05,
        text="Ruído empírico",
        showarrow=False,
    )

    fig.add_annotation(
        xref="paper",
        x=1.01,
        y=0.15,
        text="Moderado",
        showarrow=False,
    )

    fig.add_annotation(
        xref="paper",
        x=1.01,
        y=0.40,
        text="Alto",
        showarrow=False,
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Modelo",
        hovermode="closest",
        margin=dict(r=120),
    )

    return fig