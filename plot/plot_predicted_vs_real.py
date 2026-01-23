import matplotlib.pyplot as plt
import numpy as np

def plot_predicted_vs_actual(y_true, y_pred, label):

    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, s = 75, alpha=0.5, label=label)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="red",
        label="Perfect prediction"
    )

    plt.xlabel("Salário real")
    plt.ylabel("Salário previsto")
    plt.title("Predicted vs Actual — Regressão Linear")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
