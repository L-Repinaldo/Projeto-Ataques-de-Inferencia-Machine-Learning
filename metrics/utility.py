from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def compute_utility_metrics(y_train_true, y_train_pred, y_test_true, y_test_pred ):
    """
    Calcula métricas básicas de utilidade do aprendizado.
    """

    train_abs_error = abs(y_train_true - y_train_pred)
    test_abs_error  = abs(y_test_true - y_test_pred)

    mae =  mean_absolute_error(y_true = y_test_true, y_pred= y_test_pred)
    rmse = root_mean_squared_error(y_true= y_test_true, y_pred= y_test_pred)

    return {
        "train_abs_error": train_abs_error,
        "test_abs_error": test_abs_error,
        "mae": mae,
        'rmse': rmse
    }
