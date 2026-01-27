from plots import (
    plot_mean_absolute_error_X_eps,
    plot_determination_coefficient_X_eps, 
    plot_stability_X_eps,
    plot_stability_model,
    plot_mia_auc_vs_epsilon,
    plot_error_gap_vs_epsilon,
    plot_privacy_utility_tradeoff
    )


def run_plots(results, model_name):
    plot_mean_absolute_error_X_eps(results=results, model_name=model_name)
    plot_determination_coefficient_X_eps(results=results, model_name=model_name)
    plot_stability_X_eps(results=results, model_name=model_name)
    plot_stability_model(results=results, model_name=model_name)
    plot_mia_auc_vs_epsilon(results=results, model_name=model_name)
    plot_error_gap_vs_epsilon(results=results, model_name=model_name)
    plot_privacy_utility_tradeoff(results=results, model_name=model_name)