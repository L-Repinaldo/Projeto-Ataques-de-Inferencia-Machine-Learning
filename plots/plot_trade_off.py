import matplotlib.pyplot as plt
import numpy as np

def _get_by_dataset(results_df, dataset_name): 
    row = results_df[results_df["dataset"] == dataset_name]

    if row.empty:
        raise ValueError(f"Dataset {dataset_name} não encontrado nos resultados.")

    return row.iloc[0]

def _get_by_model(results, model_name): 
    df = results[results["model"] == model_name]

    if df.empty:
        raise ValueError(f"Modelo {model_name} não encontrado nos resultados.")

    return df

def plot_privacy_utility_tradeoff(utility_results, attack_results):
    """
    Trade-off direto Utilidade X VAZAMENTO(Ataque):
    X = perda relativa de utilidade (MAE)
    Y = risco de MIA (attack_acc)
    Todos os modelos no mesmo gráfico.
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    table = []
    keys = ["eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    models = utility_results['model'].unique()

    for i, model in enumerate(models):

        df_utility = _get_by_model(results= utility_results, model_name= model) 
        df_attack = _get_by_model(results= attack_results, model_name= model)


        baseline_dict = _get_by_dataset(results_df= df_utility, dataset_name= 'baseline')
        baseline_mae = baseline_dict['mae']

        utility_loss = []
        attack_acc_values = []
        labels = []

        for k in keys:
            if not k.startswith("eps_"):
                continue

            utility_k = _get_by_dataset(results_df= df_utility, dataset_name= k)
            mae = utility_k["mae"]
            loss = abs(mae - baseline_mae) / baseline_mae
            utility_loss.append(loss)

            attack_k = _get_by_dataset(results_df= df_attack, dataset_name= k)
            accuracy = attack_k["attack_acc"]
            attack_acc_values.append(accuracy)

            labels.append(f"ε={k.split('_')[1]}")
        
        utility_loss = np.array(utility_loss)
        attack_acc_values = np.array(attack_acc_values)

        plt.plot(
            utility_loss,
            attack_acc_values,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            label=model
        )

        for x, y, l in zip(utility_loss, attack_acc_values, labels):
            plt.text(x, y, l, fontsize=8, ha="left", va="bottom")

    plt.axhline(0.5, linestyle="--", color="gray", label="Ataque aleatório (~0.5)")
    plt.xlabel("Perda relativa de utilidade (MAE)")
    plt.ylabel("Risco de MIA (Attack Accuracy)")
    plt.title("Trade-off Privacidade vs Utilidade")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
