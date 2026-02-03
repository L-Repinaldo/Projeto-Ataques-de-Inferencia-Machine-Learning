import matplotlib.pyplot as plt

def plot_mean_absolute_error_X_eps(results, model_name):
    eps_values = []
    mae_values = []

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        mae = results[eps_key]["utility"]["mae"]

        eps_values.append(eps)
        mae_values.append(mae)

    baseline_mae = results["baseline"]["utility"]["mae"]

    # =======================
    # Escala dinâmica do eixo Y
    # =======================
    all_mae = mae_values + [baseline_mae]
    y_min = min(all_mae) * 0.9
    y_max = max(all_mae) * 1.1

    # =======================
    # Mapeamento de cor por nível de privacidade
    # =======================
    colors = []
    for eps in eps_values:
        if eps <= 0.2:
            colors.append("red")       # muita privacidade, baixa utilidade
        elif eps <= 1.0:
            colors.append("orange")    # zona intermediária
        else:
            colors.append("green")     # pouca privacidade, maior utilidade

    # =======================
    # Plot
    # =======================
    plt.figure(figsize=(8, 5))

    plt.scatter(eps_values, mae_values, c=colors, s=90, zorder=3)

    plt.axhline(
        y=baseline_mae,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Baseline (sem DP)"
    )

    # Rótulos dos pontos
    offset = (y_max - y_min) * 0.03
    for x, y in zip(eps_values, mae_values):
        plt.text(x, y + offset, f"{y:.0f}", ha="center", fontsize=9)

    plt.ylim(y_min, y_max)
    plt.title(f"Impacto da Privacidade Diferencial na Utilidade do Modelo -> {model_name}")
    plt.xlabel("ε (nível de privacidade)")
    plt.ylabel("Erro Médio Absoluto (MAE)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.show()
    plt.close()
