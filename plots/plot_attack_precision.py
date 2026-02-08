import matplotlib.pyplot as plt

def plot_mia_precision_advantage_vs_epsilon(results, model_name):

    eps_values = []
    precision_adv = []

    base_prec = results["baseline"]["mia"]["attack_metrics"]["attack_precision"]
    eps_values.append(0.0)
    precision_adv.append(base_prec - 0.5)

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        prec = results[eps_key]["mia"]["attack_metrics"]["attack_precision"]

        eps_values.append(eps)
        precision_adv.append(prec - 0.5)

    y_min = min(precision_adv) * 1.1
    y_max = max(precision_adv) * 1.1
    offset = (y_max - y_min) * 0.05

    colors = []
    for eps in eps_values:
        if eps == 0:
            colors.append("black")
        elif eps <= 0.2:
            colors.append("green")
        elif eps <= 1.0:
            colors.append("orange")
        else:
            colors.append("red")

    plt.figure(figsize=(8,5))

    plt.scatter(eps_values, precision_adv, c=colors, s=90, zorder=3)

    plt.axhline(
        y=0,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Precisão aleatória (vantagem = 0)"
    )

    for x, y in zip(eps_values, precision_adv):
        plt.text(x, y + offset, f"{y:.2f}", ha="center", fontsize=9)

    plt.title(f"Vantagem de Precisão do Ataque MIA → {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Precision - 0.5")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
    plt.close()
