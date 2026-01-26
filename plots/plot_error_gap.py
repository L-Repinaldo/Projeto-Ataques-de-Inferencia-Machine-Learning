import matplotlib.pyplot as plt

def plot_error_gap_vs_epsilon(results):

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    eps_values = [0.0 if k == "baseline" else float(k.split("_")[1]) for k in eps_keys]
    gaps = [results[k]["mia"]["error_gap"] for k in eps_keys]

    plt.figure(figsize=(8,5))
    plt.plot(eps_values, gaps, marker='o', color='purple')

    for x, y in zip(eps_values, gaps):
        plt.text(x, y, f"{y:.3f}", fontsize=9, ha='right')

    plt.title("Gap de erro (teste − treino) vs ε")
    plt.xlabel("ε")
    plt.ylabel("Diferença média de erro")
    plt.grid(True)
    plt.show()
