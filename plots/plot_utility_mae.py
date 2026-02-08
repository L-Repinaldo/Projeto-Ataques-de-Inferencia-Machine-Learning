import matplotlib.pyplot as plt

def plot_mean_absolute_error_X_eps(results, model_name):

    baseline_mae = results["baseline"]["utility"]["mae"]

    eps_values = []
    utility_loss_values = []

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        dp_mae = results[eps_key]["utility"]["mae"]

        utility_loss = abs(dp_mae - baseline_mae) / baseline_mae

        eps_values.append(eps)
        utility_loss_values.append(utility_loss)

    y_min = 0
    y_max = max(utility_loss_values) * 1.15
    offset = (y_max - y_min) * 0.04

    colors = []
    for eps in eps_values:
        if eps <= 0.2:
            colors.append("red")       # muita privacidade, pouca utilidade
        elif eps <= 1.0:
            colors.append("orange")    # meio termo
        else:
            colors.append("green")     # pouca privacidade, mais utilidade

    plt.figure(figsize=(8,5))

    plt.scatter(eps_values, utility_loss_values, c=colors, s=90, zorder=3)

    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Sem perda de utilidade (baseline)"
    )

    for x, y in zip(eps_values, utility_loss_values):
        plt.text(x, y + offset, f"{y:.2f}", ha="center", fontsize=9)

    plt.ylim(y_min, y_max)

    plt.title(f"Perda de Utilidade causada por DP → {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Perda relativa de utilidade (MAE)")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
    plt.close()
