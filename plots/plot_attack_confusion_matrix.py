import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_mia_confusion_matrices(results, model_name):

    keys = ["baseline"] + sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    n = len(keys)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*4.2))
    axes = axes.flatten()

    for ax, key in zip(axes, keys):

        cm = np.array(results[key]["mia"]["attack_metrics"]["attack_confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = cm / row_sums

        labels = np.empty_like(cm).astype(str)

        cell_meanings = [
            ["TN", "FP"],
            ["FN", "TP"]
        ]

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                labels[i, j] = (
                    f"{cm[i, j]}\n"
                    f"({cm_percent[i, j]*100:.1f}%)\n"
                    f"{cell_meanings[i][j]}"
                )

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Reds",
            cbar=False,
            ax=ax,
            linewidths=0.6,
            linecolor="white",
            annot_kws={"fontsize":9}
        )

        eps_label = "Baseline" if key == "baseline" else f"ε={key.split('_')[1]}"
        ax.set_title(eps_label, fontsize=11)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    for i in range(len(keys), len(axes)):
        fig.delaxes(axes[i])

    legend_text = (
        "Confusion Matrix — Membership Inference Attack\n\n"
        "TP  → True positive (vazamento)\n"
        "TN  → True Negative\n"
        "FP  → False Positive (erro do ataque)\n"
        "FN  → False Negative (proteção)\n\n"
        "Percentual é relativo à classe REAL"
    )

    fig.text(
        0.98, 0.02, legend_text,
        fontsize=10,
        va='bottom',
        ha='right',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9)
    )

    plt.suptitle(f"MIA Confusion Matrices → {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.show()
    plt.close()
