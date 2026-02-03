import matplotlib.pyplot as plt
import pandas as pd

def plot_summary_table(all_tables, title="Síntese dos Resultados"):
    """
    Recebe uma lista de dicionários com colunas iguais e gera uma tabela visual.
    """

    df = pd.DataFrame(all_tables)

    fig, ax = plt.subplots(figsize=(len(df.columns)*2.5, len(df)*0.5 + 1))
    ax.axis('off')


    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    for col in range(len(df.columns)):
        cell = table[0, col]  
        cell.set_fontsize(14) 
        cell.set_facecolor("#a6cee3")  
        cell.set_text_props(weight='bold')  

    plt.title(title, fontsize= 20)

    plt.tight_layout()
    plt.show()
    plt.close()
