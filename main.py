from data import load_data
from experiments import run_experiment

from model import (
    run_linear_regression,
    run_random_forest,
    run_elastic_net,
    run_xgboost
    )

from plots import (
    plot_mean_absolute_error_X_eps,
    plot_determination_coefficient_X_eps, 
    plot_stability_X_eps,
    plot_stability_model,
    plot_mia_auc_vs_epsilon,
    plot_error_gap_vs_epsilon,
    plot_privacy_utility_tradeoff
    )

if __name__ == "__main__":

    datasets = load_data()
    names = ["baseline", "eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    experiments = [
        ("Linear Regression", run_linear_regression),
        ("Elastic Net", run_elastic_net),
        ("XGBoost", run_xgboost),
        ("Random Forest", run_random_forest),
    ]

    for model_name, runner in experiments:

        print(f"\n{'='*40}")
        print(f"{model_name} execution")
        print(f"{'='*40}")

        exp_output = run_experiment(runner, model_name, datasets, names)
        results = exp_output["results"]

        plot_mean_absolute_error_X_eps(results, model_name)
        plot_determination_coefficient_X_eps(results, model_name)
        plot_stability_X_eps(results, model_name)
        plot_stability_model(results, model_name)
        plot_mia_auc_vs_epsilon(results, model_name)
        plot_error_gap_vs_epsilon(results, model_name)
        plot_privacy_utility_tradeoff(results, model_name)
