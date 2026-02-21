from data import load_data
from experiments import run_machine_learning_experiments, run_plots
from analysis import build_summary_table

from model import (
    run_linear_regression,
    run_random_forest,
    run_elastic_net,
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
                    "accuracy": payload["results"]["attack_accuracy"],
                    "f1": payload["results"]["attack_f1_score"],
                    "precision": payload["results"]["attack_precision"],
                    "auc": payload["results"]["attack_roc_auc"],
                })

            #TODO ajustar run_plots, plots e build_summary_table

            #run_plots(results= experiment_results, model_name= model_name)

            #summary = build_summary_table(results= experiment_results[model_name])
            #all_tables.extend(summary)
    
    df_utility = pd.DataFrame(utility_rows)
    df_attack = pd.DataFrame(attack_rows)

    from plots import plot_summary_table

    plot_summary_table(all_tables= all_tables)
