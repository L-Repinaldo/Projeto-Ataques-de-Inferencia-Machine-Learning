import matplotlib.pyplot as plt

def plot_stability_X_eps(results):
    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    eps_labels = []
    mae_cv = []
    r2_cv = []

    for k in eps_keys:
        eps_labels.append("0" if k == "baseline" else k.split("_")[1])
        stability = results[k].get("stability", {})
        mae_cv.append(stability.get("mae_cv", 0.0))
        r2_cv.append(stability.get("r2_cv", 0.0))

    # Cores para cada ponto
    colors_mae = ["gray", "darkblue", "blue", "deepskyblue", "lightblue"][:len(mae_cv)]
    colors_r2  = ["gray", "darkgreen", "green", "limegreen", "lightgreen"][:len(r2_cv)]

    plt.figure(figsize=(8,9))

    # Plot MAE CV
    for x, y, c in zip(eps_labels, mae_cv, colors_mae):
        plt.scatter(x, y, color=c, s=80)
        plt.text(x, y +  0.1*max(r2_cv), f"{y:.2f}", ha='center', fontsize=9)
    plt.plot(eps_labels, mae_cv, color="blue", linestyle="--", alpha=0.5)

    # Plot R² CV
    for x, y, c in zip(eps_labels, r2_cv, colors_r2):
        plt.scatter(x, y, color=c, s=80, marker='s')
        plt.text(x, y + 0.03, f"{y:.2f}", ha='center', fontsize=9)
    plt.plot(eps_labels, r2_cv, color="green", linestyle="--", alpha=0.5)

    plt.title("Instabilidade relativa por ε (CV)")
    plt.xlabel("ε")
    plt.ylabel("Coeficiente de variação (CV)")

    # Ajuste automático do ylim
    y_max = max(max(mae_cv), max(r2_cv)) * 1.2  # 20% de folga
    plt.ylim(0, y_max)
    
    plt.grid(True)
    plt.legend(["MAE CV", "R² CV"])
    plt.show()



def plot_stability_model(results):
    metrics = ["MAE", "R²"]
    cv_values = [
        results["stability_mae"]["cv"],
        results["stability_r2"]["cv"]
    ]

    colors = ["blue", "green"]

    plt.figure(figsize=(6,5))
    plt.bar(metrics, cv_values, color=colors, alpha=0.7)
    
    for i, v in enumerate(cv_values):
        plt.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=12)

    plt.ylim(0,1)
    plt.title("Instabilidade relativa (CV) agregada")
    plt.ylabel("Coeficiente de variação (CV)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
