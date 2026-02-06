from data import load_data
from experiments import run_experiment, run_plots
from analysis import build_summary_table

from model import (
    run_linear_regression,
    run_random_forest,
    run_elastic_net,
    run_xgboost
    )

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
        ("Linear Regression", run_linear_regression),
        ("Elastic Net", run_elastic_net),
        ("XGBoost", run_xgboost),
        ("Random Forest", run_random_forest),
    ]

    all_tables = []

    for model_name, runner in experiments:

        print(f"\n{'='*40}")
        print(f"{model_name} execution")
        print(f"{'='*40}")

        exp_output = run_experiment(runner, model_name, datasets, names)
        results = exp_output["results"]

        run_plots(results= results, model_name= model_name)

        summary = build_summary_table(results, model_name)
        all_tables.extend(summary)


    from plots import plot_summary_table

    plot_summary_table(all_tables= all_tables)

