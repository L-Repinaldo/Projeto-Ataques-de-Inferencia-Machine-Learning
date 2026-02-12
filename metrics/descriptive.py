def compute_descriptive_metrics(info_baseline, df):
    """
    Calcula métricas básicas de estatística descritiva.
    """

    baseline_mean = info_baseline['mean']
    baseline_median = info_baseline['median']

    df_mean = df.mean()
    df_median = df.median()

    mean_loss = df_mean - baseline_mean / baseline_mean
    median_loss = df_median - baseline_median / baseline_median

    return { 
        "mean_loss" : mean_loss,
        "median_loss" : median_loss
    }