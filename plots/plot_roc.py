import matplotlib.pyplot as plt

def plot_mia_roc_all_eps(results, model_name):

    plt.figure(figsize=(7,7))

    fpr, tpr = results["baseline"]["mia"]["attack_metrics"]["attack_roc_curve"]
    auc = results["baseline"]["mia"]["attack_metrics"]["attack_roc_auc"]
    plt.plot(fpr, tpr, linewidth=2, label=f"baseline (AUC={auc:.2f})")

    eps_keys = sorted(
        [k for k in results.keys() if k.startswith("eps_")],
        key=lambda x: float(x.split("_")[1])
    )

    for k in eps_keys:
        eps = float(k.split("_")[1])
        fpr, tpr = results[k]["mia"]["attack_metrics"]["attack_roc_curve"]
        auc = results[k]["mia"]["attack_metrics"]["attack_roc_auc"]

        plt.plot(fpr, tpr, linestyle="--", label=f"ε={eps} (AUC={auc:.2f})")

    plt.plot([0,1], [0,1], 'k--', alpha=0.5, label="Aleatório")

    plt.xlabel("Taxa de Falso Positivo (FPR)")
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
    plt.title(f"Curvas ROC do Ataque de Membership Inference — {model_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.show()
    plt.close()
