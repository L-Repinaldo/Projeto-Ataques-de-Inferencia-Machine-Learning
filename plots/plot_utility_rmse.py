import matplotlib.pyplot as plt

def plot_rmse_utility_loss_vs_epsilon(results, model_name):

    baseline_rmse = results["baseline"]["utility"]["rmse"]

    eps_values = []
    utility_loss_values = []

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        dp_rmse = results[eps_key]["utility"]["rmse"]

        utility_loss = abs(dp_rmse - baseline_rmse) / baseline_rmse

        eps_values.append(eps)
        utility_loss_values.append(utility_loss)

    y_min = 0
    y_max = max(utility_loss_values) * 1.15 if utility_loss_values else 1
    offset = (y_max - y_min) * 0.04

    colors = []
    for eps in eps_values:
        if eps <= 0.2:
            colors.append("red")       # muita privacidade
        elif eps <= 1.0:
            colors.append("orange")    # intermediário
        else:
            colors.append("green")     # pouca privacidade

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

    plt.title(f"Perda de Utilidade (RMSE) causada por DP → {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Perda relativa de utilidade (RMSE)")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
    plt.close()
