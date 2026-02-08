import matplotlib.pyplot as plt

def plot_mia_accuracy_advantage_vs_epsilon(results, model_name):

    eps_values = []
    advantage_values = []

    baseline_acc = results["baseline"]["mia"]["attack_metrics"]["attack_accuracy"]
    eps_values.append(0.0)
    advantage_values.append(baseline_acc - 0.5)

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        acc = results[eps_key]["mia"]["attack_metrics"]["attack_accuracy"]

        eps_values.append(eps)
        advantage_values.append(acc - 0.5)

    y_min = min(advantage_values) * 1.1
    y_max = max(advantage_values) * 1.1
    offset = (y_max - y_min) * 0.05

    colors = []
    for eps in eps_values:
        if eps == 0:
            colors.append("black")
        elif eps <= 0.2:
            colors.append("green")    # mais privado
        elif eps <= 1.0:
            colors.append("orange")
        else:
            colors.append("red")      # pouco privado

    plt.figure(figsize=(8,5))

    plt.scatter(eps_values, advantage_values, c=colors, s=90, zorder=3)

    plt.axhline(
        y=0,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Ataque aleatório (vantagem = 0)"
    )

    for x, y in zip(eps_values, advantage_values):
        plt.text(x, y + offset, f"{y:.2f}", ha="center", fontsize=9)

    plt.title(f"Vantagem do Ataque MIA (Accuracy) sob DP → {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Vantagem do atacante (Accuracy - 0.5)")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
    plt.close()
