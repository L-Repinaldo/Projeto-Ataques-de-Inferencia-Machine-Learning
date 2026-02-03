import matplotlib.pyplot as plt
import numpy as np

def plot_privacy_utility_tradeoff(results, model_name):

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    mae_values = np.array([results[k]["utility"]["mae"] for k in eps_keys])
    auc_values = np.array([results[k]["mia"]["attack_auc"] for k in eps_keys])
    labels = ["baseline"] + [k.split("_")[1] for k in eps_keys if k != "baseline"]

    plt.figure(figsize=(7,6))

    # Scatter principal
    plt.scatter(mae_values, auc_values, s=110, c="darkred")

    # Linha de ataque aleatório
    plt.axhline(0.5, linestyle='--', color='gray', label="Ataque aleatório (AUC=0.5)")

    x_range = mae_values.max() - mae_values.min()
    y_range = auc_values.max() - auc_values.min()

    x_margin = max(x_range * 0.15, mae_values.mean() * 0.05)
    y_margin = max(y_range * 0.15, 0.05)

    plt.xlim(mae_values.min() - x_margin, mae_values.max() + x_margin)
    plt.ylim(max(0.4, auc_values.min() - y_margin), min(1.0, auc_values.max() + y_margin))

    # Rótulos dos pontos
    for x, y, l in zip(mae_values, auc_values, labels):
        plt.text(x, y, l, fontsize=9, ha='left', va='bottom')

    plt.text(0.02, 0.98,
         "↓ MAE = melhor utilidade",
         transform=plt.gca().transAxes,
         fontsize=9,
         verticalalignment='top')

    plt.text(0.02, 0.93,
         "↓ AUC → ataque próximo do aleatório (mais seguro)",
         transform=plt.gca().transAxes,
         fontsize=9,
         verticalalignment='top')

    plt.title(f"Trade-off entre Utilidade e Risco de Membership Inference -> {model_name}")
    plt.xlabel("MAE (↑ pior utilidade)")
    plt.ylabel("AUC do ataque (↑ maior risco)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
