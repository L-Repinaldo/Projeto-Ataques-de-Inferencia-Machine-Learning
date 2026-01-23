from sklearn.metrics import mean_absolute_error, r2_score

def compute_utility_metrics(y_true, y_pred):
    """
    Calcula métricas básicas de utilidade do aprendizado.

    Retorna métricas simples, interpretáveis e alinhadas ao README.
    """

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred), #Coeficiente de determinação

    }
