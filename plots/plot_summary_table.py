import matplotlib.pyplot as plt
import pandas as pd

def plot_summary_table(all_tables, title="Síntese dos Resultados" ):
    """
    Recebe uma lista de dicionários com colunas iguais e gera uma tabela visual.
    """

    df = pd.DataFrame(all_tables).sort_values("modelo")

    fig, ax = plt.subplots(figsize=(len(df.columns)*2.5, len(df)*0.6 + 1))
    ax.axis('off')


    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    for col in range(len(df.columns)):
        cell = table[0, col]  
        cell.set_fontsize(14) 
        cell.set_facecolor("#a6cee3")  
        cell.set_text_props(weight='bold', color= 'white')  

    util_colors = {
        "pouco afetada": "#b6e3b6",
        "moderadamente afetada": "#ffe599",
        "muito afetada": "#f4cccc"
    }

    risk_colors = {
        "quase aleatório": "#b6e3b6",
        "vazamento fraco": "#fff2cc",
        "vazamento moderado": "#f9cb9c",
        "vazamento forte": "#ea9999"
    }

    for row in range(len(df)):
        util_cell = table[row+1, df.columns.get_loc("usabilidade")]
        risk_cell = table[row+1, df.columns.get_loc("seguranca")]

        util_cell.set_facecolor(util_colors.get(df.iloc[row]["usabilidade"], "white"))
        risk_cell.set_facecolor(risk_colors.get(df.iloc[row]["seguranca"], "white"))

    plt.title(title, fontsize= 20)

    plt.tight_layout()
    plt.show()
    plt.close()
