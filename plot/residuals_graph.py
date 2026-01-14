import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(residuals, name):
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)

    plt.title(f"Distribuição dos erros — {name}")
    plt.xlabel("Erro (salário real - previsto)")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.show()
