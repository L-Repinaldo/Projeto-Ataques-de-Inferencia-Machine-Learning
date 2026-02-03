import matplotlib.pyplot as plt

def plot_stability_X_eps(results, model_name):
    eps_values = []
    mae_cv = []
    r2_cv = []

    # baseline
    eps_values.append(0.0)
    mae_cv.append(results["baseline"]["stability"]["mae_cv"])
    r2_cv.append(results["baseline"]["stability"]["r2_cv"])

    for eps_key in sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    ):
        eps = float(eps_key.split("_")[1])
        eps_values.append(eps)
        mae_cv.append(results[eps_key]["stability"]["mae_cv"])
        r2_cv.append(results[eps_key]["stability"]["r2_cv"])

    # =======================
    # Escalas dinâmicas
    # =======================
    mae_min, mae_max = min(mae_cv), max(mae_cv)
    r2_min, r2_max = min(r2_cv), max(r2_cv)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # -------- MAE CV --------
    axes[0].scatter(eps_values, mae_cv, color="royalblue", s=90)
    offset_mae = (mae_max - mae_min) * 0.05
    for x, y in zip(eps_values, mae_cv):
        axes[0].text(x, y + offset_mae, f"{y:.2f}", ha="center", fontsize=9)

    axes[0].set_ylabel("MAE CV")
    axes[0].set_title(f"Instabilidade do Modelo sob Privacidade Diferencial -> {model_name}")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylim(mae_min - offset_mae, mae_max + offset_mae)

    # -------- R² CV --------
    axes[1].scatter(eps_values, r2_cv, color="seagreen", s=90)
    offset_r2 = (r2_max - r2_min) * 0.05
    for x, y in zip(eps_values, r2_cv):
        axes[1].text(x, y + offset_r2, f"{y:.2f}", ha="center", fontsize=9)

    axes[1].set_xlabel("ε (nível de privacidade)")
    axes[1].set_ylabel("R² CV")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_ylim(r2_min - offset_r2, r2_max + offset_r2)

    plt.show()



def plot_stability_model(results, model_name):
    metrics = ["MAE CV", "R² CV"]
    cv_values = [
        results["stability_mae"]["cv"],
        results["stability_r2"]["cv"]
    ]

    y_min = min(cv_values)
    y_max = max(cv_values)
    offset = (y_max - y_min) * 0.1

    plt.figure(figsize=(6,5))
    bars = plt.bar(metrics, cv_values, color=["royalblue", "seagreen"], alpha=0.75)

    for bar, v in zip(bars, cv_values):
        plt.text(bar.get_x() + bar.get_width()/2, v + offset*0.2,
                 f"{v:.2f}", ha='center', fontsize=12)

    plt.ylim(y_min - offset, y_max + offset)
    plt.title(f"Instabilidade Relativa Global do Modelo -> {model_name}")
    plt.ylabel("Coeficiente de Variação (CV)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()
    plt.close()
