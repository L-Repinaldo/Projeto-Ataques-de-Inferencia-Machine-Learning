import pandas as pd
import plotly.graph_objects as go


UTILITY_COLORS = {
    "pouco afetada": "#b6e3b6",
    "moderadamente afetada": "#ffe599",
    "muito afetada": "#f4cccc",
}

RISK_COLORS = {
    "pior que aleatório": "#d9d9d9",
    "quase aleatório": "#e8f5e9",
    "vazamento muito fraco": "#c8e6c9",
    "vazamento fraco": "#fff9c4",
    "vazamento moderado": "#ffe0b2",
    "vazamento alto": "#ffab91",
    "vazamento muito alto": "#e57373",
}


def _column_fill_colors(df):
    colors = []

    for column in df.columns:
        if column == "utilidade":
            colors.append([UTILITY_COLORS.get(value, "white") for value in df[column]])
        elif column == "vazamento":
            colors.append([RISK_COLORS.get(value, "white") for value in df[column]])
        else:
            colors.append(["white"] * len(df))

    return colors


def plot_summary_table(all_tables, title="Síntese dos Resultados"):
    """
    Recebe uma lista de dicionários com colunas iguais e gera uma tabela visual.
    """
    df = pd.DataFrame(all_tables).sort_values("modelo")

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(df.columns),
                    "fill_color": "#a6cee3",
                    "font": {"color": "white", "size": 14},
                    "align": "center",
                },
                cells={
                    "values": [df[column] for column in df.columns],
                    "fill_color": _column_fill_colors(df),
                    "font": {"color": "#222222", "size": 12},
                    "align": "center",
                    "height": 28,
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.show()
    return fig
