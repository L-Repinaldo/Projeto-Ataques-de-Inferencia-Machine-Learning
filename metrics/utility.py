from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def compute_utility_metrics(y_true, y_pred):
    """
    Calcula métricas básicas de utilidade do aprendizado.
    """

    mae =  mean_absolute_error(y_true = y_true, y_pred= y_pred)
    rmse = root_mean_squared_error(y_true= y_true, y_pred= y_pred)

    return {
        "mae": mae,
        'rmse': rmse
    }
