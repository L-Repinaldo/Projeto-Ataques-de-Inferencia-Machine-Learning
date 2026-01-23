import numpy as np

def compute_stability(metric_values):
    """
    Mede estabilidade como variabilidade entre execuções independentes.

    Espera uma lista de valores da mesma métrica (ex: MAE ou R²)
    obtidos sob o mesmo ε.
    """

    values = np.array(metric_values)

    return {
        "mean": values.mean(),
        "std": values.std(), #Desvio Padrão
        "cv": values.std() / values.mean() if values.mean() != 0 else None, #Coeficiente de variação
    }
