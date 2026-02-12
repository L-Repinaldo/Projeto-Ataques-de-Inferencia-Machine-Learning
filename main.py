from data import load_data
from experiments import run_machine_learning_experiments, run_statistic_experiment, run_plots
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

    experiment_results = {}

    for model_name, runner in experiments:

        print(f"\n{'='*40}")
        print(f"{model_name} execution")
        print(f"{'='*40}")

        model_metric_results, attack_metrics_results = run_machine_learning_experiments(model_runner= runner, model_name= model_name, 
                                                                                                    datasets= datasets,dataset_names= names)
        experiment_results[model_name] = {}

        experiment_results[model_name]['Metrics Model'] = model_metric_results
        experiment_results[model_name]['Metrics Attack'] = attack_metrics_results

        run_plots(results= experiment_results, model_name= model_name)

        summary = build_summary_table(results= experiment_results[model_name])
        all_tables.extend(summary)

    print(f"\n{'='*40}")
    print(f"Descriptive Statistic Loss execution")
    print(f"{'='*40}")
    descriptive_statistic_results = run_statistic_experiment(datasets= datasets, dataset_names= names)
    experiment_results[model_name]['Descriptive Statistic'] = descriptive_statistic_results


        


        


    from plots import plot_summary_table

    plot_summary_table(all_tables= all_tables)


