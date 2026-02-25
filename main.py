from data import load_data
from experiments import run_machine_learning_experiments, run_plots
from analysis import build_summary_table

from model import (
    run_random_forest,
    run_xgboost
    )

import pandas as pd

if __name__ == "__main__":

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn.preprocessing._encoders"
        )

    datasets = load_data()
    names = ["baseline", "eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    experiments = [
        ("XGBoost", run_xgboost),
        ("Random Forest", run_random_forest),
    ]

    all_tables = []

    utility_rows = []
    attack_rows = []

    for model_name, runner in experiments:

        print(f"\n{'='*40}")
        print(f"{model_name} execution")
        print(f"{'='*40}")

        model_metric_results, attack_metrics_results = run_machine_learning_experiments(model_runner= runner, model_name= model_name, 
                                                                                                    datasets= datasets,dataset_names= names)
        

        for dataset_name, payload in model_metric_results.items():
            
            utility_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "mae": payload["results"]["mae"],
                "rmse": payload["results"]["rmse"],
            })

        for dataset_name, payload in attack_metrics_results.items():

            attack_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "attack_acc": payload["results"]["attack_acc"],
                "member_acc": payload["results"]["member_acc"],
                "non_member_acc": payload["results"]["non_member_acc"],
                "precision": payload["results"]["precision"],
                "recall": payload["results"]["recall"],
            })
    
    df_utility = pd.DataFrame(utility_rows)
    df_attack = pd.DataFrame(attack_rows)

    summary = build_summary_table(utility_results= df_utility, attack_results= df_attack)
    #TODO ajustar run_plots, plots

    #run_plots(results= experiment_results, model_name= model_name)

    from plots import plot_summary_table

    plot_summary_table(all_tables= summary)
