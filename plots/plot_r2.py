import matplotlib.pyplot as plt

def plot_determination_coefficient_X_eps(results, model_name):
    eps_values = []
    r2_values = []

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        r2 = results[eps_key]["utility"]["r2"]

        eps_values.append(eps)
        r2_values.append(r2)

    baseline_r2 = results["baseline"]["utility"]["r2"]

    # =======================
    # Escala dinâmica
    # =======================
    all_r2 = r2_values + [baseline_r2]
    y_min = min(all_r2) - abs(min(all_r2))*0.1
    y_max = max(all_r2) + abs(max(all_r2))*0.1

    # =======================
    # Cores por regime de privacidade
    # =======================
    colors = []
    for eps in eps_values:
        if eps <= 0.2:
            colors.append("red")       # muita privacidade → utilidade tende a cair
        elif eps <= 1.0:
            colors.append("orange")
        else:
            colors.append("green")

    # =======================
    # Plot
    # =======================
    plt.figure(figsize=(8, 5))

    plt.scatter(eps_values, r2_values, c=colors, s=90, zorder=3)

    # Linha baseline
    plt.axhline(
        y=baseline_r2,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Baseline (sem DP)"
    )

    # Rótulos
    offset = (y_max - y_min) * 0.03
    for x, y in zip(eps_values, r2_values):
        plt.text(x, y + offset, f"{y:.2f}", ha="center", fontsize=9)

    plt.ylim(y_min, y_max)
    plt.title(f"Impacto da Privacidade Diferencial no Poder Explicativo do Modelo -> {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("R² (Coeficiente de Determinação)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
