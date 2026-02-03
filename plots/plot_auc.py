import matplotlib.pyplot as plt

def plot_mia_auc_vs_epsilon(results, model_name):

    eps_values = []
    auc_values = []

    eps_values.append(0.0)
    auc_values.append(results["baseline"]["mia"]["attack_auc"])

    for k in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps_values.append(float(k.split("_")[1]))
        auc_values.append(results[k]["mia"]["attack_auc"])

    y_min = min(auc_values)
    y_max = max(auc_values)
    offset = (y_max - y_min) * 0.05

    plt.figure(figsize=(8,5))

    plt.scatter(eps_values, auc_values, color="crimson", s=90, zorder=3)

    for x, y in zip(eps_values, auc_values):
        plt.text(x, y + offset, f"{y:.2f}", ha='center', fontsize=9)

    plt.axhline(0.5, linestyle='--', color='gray', label="Ataque aleatório (AUC = 0.5)")
    plt.scatter([], [], color="crimson", label="AUC > 0.5 indica sinal explorável")

    plt.title(f"Risco de Membership Inference sob Privacidade Diferencial -> {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("AUC do Ataque de Inferência")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(y_min - offset, y_max + offset)
    plt.legend()

    plt.show()
    plt.close()
