import matplotlib.pyplot as plt
import numpy as np

def plot_privacy_utility_tradeoff(results, model_name):

    baseline_mae = results["baseline"]["utility"]["mae"]

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    utility_loss = []
    auc_values = []
    labels = []

    for k in eps_keys:
        mae = results[k]["utility"]["mae"]
        auc = results[k]["mia"]["attack_metrics"]["attack_roc_auc"]

        loss = abs(mae - baseline_mae) / baseline_mae
        utility_loss.append(loss)
        auc_values.append(auc)

        labels.append("baseline" if k == "baseline" else f"ε={k.split('_')[1]}")

    utility_loss = np.array(utility_loss)
    auc_values = np.array(auc_values)

    plt.figure(figsize=(7,6))

    plt.scatter(utility_loss, auc_values, s=110, c="darkred")

    plt.axhline(0.5, linestyle='--', color='gray', label="Ataque aleatório (AUC=0.5)")

    for x, y, l in zip(utility_loss, auc_values, labels):
        plt.text(x, y, l, fontsize=9, ha='left', va='bottom')

    plt.title(f"Trade-off Privacidade vs Utilidade → {model_name}")
    plt.xlabel("Perda relativa de utilidade (MAE)")
    plt.ylabel("Risco de MIA (AUC)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
