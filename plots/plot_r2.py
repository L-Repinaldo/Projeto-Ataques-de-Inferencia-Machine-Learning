import matplotlib.pyplot as plt

def plot_determination_coefficient_X_eps(results):
    eps_labels = []
    r2_values = []

    if "baseline" in results:
        eps_labels.append("baseline")
        r2_values.append(results["baseline"]["utility"]["r2"])

    eps_keys = sorted([k for k in results.keys() if k.startswith("eps_")],
                      key=lambda x: float(x.split("_")[1]))
    for k in eps_keys:
        eps_labels.append(k.split("_")[1])
        r2_values.append(results[k]["utility"]["r2"])

    colors = ["gray"] + ["darkgreen", "green", "limegreen", "lightgreen"][:len(r2_values)-1]

    plt.figure(figsize=(8,5))
    for x, y, c in zip(eps_labels, r2_values, colors):
        plt.scatter(x, y, color=c, s=80)
        plt.text(x, y + 0.03, f"{y:.2f}", ha='center', fontsize=9)

    plt.plot(eps_labels, r2_values, color="green", linestyle="--", alpha=0.5)

    plt.title("Coeficiente de Determinação (R²) vs ε")
    plt.xlabel("ε")
    plt.ylabel("R²")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()
