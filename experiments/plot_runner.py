from plots import plot_tables_chart, plot_privacy_utility_tradeoff
    


def run_plots(df_utility, df_attack):

    ############
    # Utility
    ############

    plot_tables_chart(results= df_utility, title= "Metricas utilidade " )

    ############
    # Attack
    ############

    plot_tables_chart(results= df_attack, title= "Metricas ataque " )


    ############
    # Trade off
    ############
    plot_privacy_utility_tradeoff(utility_results= df_utility, attack_results= df_attack)