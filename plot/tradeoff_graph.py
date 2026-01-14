import matplotlib.pyplot as plt

def plot_tradeoff(dict_rmse):

    baseline_rmse = dict_rmse['baseline']

    dict_rmse.pop('baseline')

    eps = [float(k.split("_")[1]) for k in dict_rmse.keys()]
    rmse = list(dict_rmse.values())


    plt.figure(figsize=(8, 5))

    plt.plot(eps, rmse, marker="o", label="DP datasets")
    plt.axhline(
        y = baseline_rmse,
        linestyle = "--",
        color = "red",
        label = "Baseline"
    )

    plt.xlabel("Epsilon (ε)")
    plt.ylabel("RMSE")
    plt.title("Trade-off entre Privacidade Diferencial e Utilidade")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

