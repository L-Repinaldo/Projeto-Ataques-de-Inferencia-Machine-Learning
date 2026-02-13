from plots import (
    plot_mean_absolute_error_X_eps,
    plot_rmse_utility_loss_vs_epsilon,
    plot_mia_confusion_matrices,
    plot_mia_accuracy_advantage_vs_epsilon,
    plot_mia_precision_advantage_vs_epsilon,
    plot_mia_f1_advantage_vs_epsilon,
    plot_mia_auc_vs_epsilon,
    plot_mia_roc_all_eps,
    plot_privacy_utility_tradeoff,
    )


def run_plots(results, model_name):

    ############
    # Utility
    ############

    model_utility_results = results[model_name]['Metrics Model']

    plot_mean_absolute_error_X_eps(results= model_utility_results, model_name=model_name)
    plot_rmse_utility_loss_vs_epsilon(results= model_utility_results, model_name= model_name)

    ############
    # Attack
    ############

    attack_results = results[model_name]['Metrics Attack']

    plot_mia_confusion_matrices(results= attack_results, model_name= model_name)
    plot_mia_accuracy_advantage_vs_epsilon(results= attack_results, model_name= model_name)
    plot_mia_precision_advantage_vs_epsilon(results= attack_results, model_name = model_name)
    plot_mia_f1_advantage_vs_epsilon(results= attack_results, model_name= model_name)
    plot_mia_roc_all_eps(results= attack_results, model_name= model_name)
    plot_mia_auc_vs_epsilon(results= attack_results, model_name=model_name)

    ############
    # Trade off
    ############
    plot_privacy_utility_tradeoff(model_utility_results= model_utility_results, attack_results= attack_results, model_name=model_name)