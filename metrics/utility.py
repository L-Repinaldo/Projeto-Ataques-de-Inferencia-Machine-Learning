from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def compute_utility_metrics(prediction_result):
    """
    Calcula métricas básicas de utilidade do aprendizado.
    """

    train_abs_error = abs(prediction_result.y_train_true - prediction_result.y_train_pred)
    test_abs_error = abs(prediction_result.y_test_true - prediction_result.y_test_pred)

    mae = mean_absolute_error(
        y_true=prediction_result.y_test_true,
        y_pred=prediction_result.y_test_pred,
    )
    rmse = root_mean_squared_error(
        y_true=prediction_result.y_test_true,
        y_pred=prediction_result.y_test_pred,
    )

    return {
        "train_abs_error": train_abs_error,
        "test_abs_error": test_abs_error,
        "mae": mae,
        "rmse": rmse,
    }
