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
        "pior que aleatório": "#d9d9d9",      
        "quase aleatório": "#e8f5e9",         
        "vazamento muito fraco": "#c8e6c9",   
        "vazamento fraco": "#fff9c4",         
        "vazamento moderado": "#ffe0b2",      
        "vazamento alto": "#ffab91",          
        "vazamento muito alto": "#e57373"     
    }

    for row in range(len(df)):
        util_cell = table[row+1, df.columns.get_loc("utilidade")]
        risk_cell = table[row+1, df.columns.get_loc("vazamento")]

        util_cell.set_facecolor(util_colors.get(df.iloc[row]["utilidade"], "white"))
        risk_cell.set_facecolor(risk_colors.get(df.iloc[row]["vazamento"], "white"))

    plt.title(title, fontsize= 20)

    plt.tight_layout()
    plt.show()
    plt.close()
