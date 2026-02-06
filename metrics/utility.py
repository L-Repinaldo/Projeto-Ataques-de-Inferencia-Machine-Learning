from sklearn.metrics import mean_absolute_error

def compute_utility_metrics(y_true, y_pred):
    """
    Calcula métricas básicas de utilidade do aprendizado.
    """

    mae =  mean_absolute_error(y_true = y_true, y_pred= y_pred)

    return {
        "mae": mae,
    }
