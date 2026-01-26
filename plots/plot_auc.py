
import matplotlib.pyplot as plt

def plot_mia_auc_vs_epsilon(results):

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    eps_values = [0.0 if k == "baseline" else float(k.split("_")[1]) for k in eps_keys]
    auc_values = [results[k]["mia"]["attack_auc"] for k in eps_keys]

    plt.figure(figsize=(8,5))
    plt.plot(eps_values, auc_values, marker='o', color='red')

    for x, y in zip(eps_values, auc_values):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha='right')

    plt.axhline(0.5, linestyle='--', color='gray', label="Ataque aleatório")

    plt.title("Risco de Membership Inference vs ε")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Area Under Curve do ataque")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()

