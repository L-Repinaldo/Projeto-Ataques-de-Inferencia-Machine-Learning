import matplotlib.pyplot as plt

def plot_mean_absolute_error_X_eps(results):
    # Separar baseline
    mae_values = []
    eps_labels = []

    if "baseline" in results:
        eps_labels.append("baseline")
        mae_values.append(results["baseline"]["utility"]["mae"])

    # Agora adiciona os demais eps de forma ordenada
    for eps_key in sorted([k for k in results.keys() if k.startswith("eps_")], 
                          key=lambda x: float(x.split("_")[1])):
        eps_labels.append(results[eps_key].get("label", eps_key))
        mae_values.append(results[eps_key]["utility"]["mae"])

    colors = ["gray"] + ["blue", "deepskyblue", "skyblue", "lightblue"][:len(mae_values)-1]

    plt.figure(figsize=(8,5))
    for x, y, c in zip(eps_labels, mae_values, colors):
        plt.scatter(x, y, color=c, s=80)
        plt.text(x, y + y*0.03, f"{y:.0f}", ha='center', fontsize=9)


    plt.plot(eps_labels, mae_values, color="blue", linestyle="--", alpha=0.5)

    plt.title("Erro Médio Absoluto (MAE) vs ε")
    plt.xlabel("ε")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.show()

