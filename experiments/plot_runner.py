from plots import (
    plot_mean_absolute_error_X_eps,
    plot_mia_auc_vs_epsilon,
    plot_privacy_utility_tradeoff,
    plot_mia_roc_all_eps
    )


def run_plots(results, model_name):
    plot_mean_absolute_error_X_eps(results=results, model_name=model_name)
    plot_mia_roc_all_eps(results= results, model_name= model_name)
    plot_mia_auc_vs_epsilon(results=results, model_name=model_name)
    plot_privacy_utility_tradeoff(results=results, model_name=model_name)