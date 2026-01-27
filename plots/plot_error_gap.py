import matplotlib.pyplot as plt
import numpy as np

def plot_error_gap_vs_epsilon(results, model_name):

    eps_keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    eps_values = [0.0 if k == "baseline" else float(k.split("_")[1]) for k in eps_keys]
    gaps = [results[k]["mia"]["error_gap"] for k in eps_keys]

    plt.figure(figsize=(8,5))

    plt.plot(eps_values, gaps, marker='o', color='purple')

    plt.axhline(0, linestyle='--', color='gray', 
                label="Gap = 0 (mesmo erro treino/teste)")

    g_min, g_max = min(gaps), max(gaps)
    margin = (g_max - g_min) * 0.15 if g_max != g_min else 0.1
    plt.ylim(g_min - margin, g_max + margin)

    for x, y in zip(eps_values, gaps):
        offset = margin * 0.15
        plt.text(x, y + offset, f"{y:.3f}", fontsize=9, ha='center')

    plt.title(f"Gap de erro (teste - treino) vs ε -> {model_name}")
    plt.xlabel("ε")
    plt.ylabel("Diferença média de erro")

    plt.grid(True)
    plt.legend()
    plt.show()
