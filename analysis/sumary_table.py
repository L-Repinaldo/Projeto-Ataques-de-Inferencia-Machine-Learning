from .impact_classifier import classify_utility_impact, classify_security_risk

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

def build_summary_table(utility_results, attack_results):

    table = []
    keys = ["baseline", "eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    models = utility_results['model'].unique()

    for model in models:

        df_utility = _get_by_model(results= utility_results, model_name= model) 
        df_attack = _get_by_model(results= attack_results, model_name= model)


        baseline_dict = _get_by_dataset(results_df= df_utility, dataset_name= 'baseline')
        baseline_mae = baseline_dict['mae']

        for k in keys:
            if not k.startswith("eps_"):
                continue

            utility_k = _get_by_dataset(results_df= df_utility, dataset_name= k)
            mae = utility_k["mae"]
            mae_cv = abs(mae - baseline_mae) / baseline_mae

            attack_k = _get_by_dataset(results_df= df_attack, dataset_name= k)
            accuracy = attack_k["attack_acc"]

            row = {
                "modelo": utility_k["model"],
                "epsilon": float(k.split("_")[1]),
                "usabilidade": classify_utility_impact(mae_cv),
                "seguranca": classify_security_risk(accuracy),
            }

            table.append(row)

    return table