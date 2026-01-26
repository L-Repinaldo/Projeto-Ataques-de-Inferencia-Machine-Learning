import matplotlib.pyplot as plt

def plot_privacy_utility_tradeoff(results):

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    mae_values = [results[k]["utility"]["mae"] for k in eps_keys]
    auc_values = [results[k]["mia"]["attack_auc"] for k in eps_keys]

    labels = ["baseline"] + [k.split("_")[1] for k in eps_keys if k != "baseline"]

    plt.figure(figsize=(7,6))
    plt.scatter(mae_values, auc_values, s=100)

    for x, y, l in zip(mae_values, auc_values, labels):
        plt.text(x, y, l, fontsize=9)

    plt.title("Trade-off Privacidade vs Utilidade")
    plt.xlabel("MAE (↑ pior utilidade)")
    plt.ylabel("AUC do ataque (↑ maior risco)")
    plt.grid(True)
    plt.show()
