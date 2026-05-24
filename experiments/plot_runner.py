from analysis import build_summary_table
from visualization import (
    plot_attack_metrics,
    plot_privacy_utility_tradeoff,
    plot_summary_table,
    plot_utility_metrics,
)
from visualization.common import load_metric_artifacts


def run_plots(df_utility, df_attack, summary=None):

    ############
    # Utility
    ############

    plot_utility_metrics(df_utility=df_utility)

    ############
    # Attack
    ############

    plot_attack_metrics(df_attack=df_attack)


    ############
    # Trade off
    ############
    plot_privacy_utility_tradeoff(utility_results=df_utility, attack_results=df_attack)

    if summary is not None:
        plot_summary_table(all_tables=summary)


def run_plots_from_artifacts(experiment_dir):
    df_utility, df_attack = load_metric_artifacts(experiment_dir)
    summary = build_summary_table(utility_results=df_utility, attack_results=df_attack)
    run_plots(df_utility=df_utility, df_attack=df_attack, summary=summary)
