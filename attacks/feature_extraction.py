def extract_attack_features(utility_metrics=None, model_output=None):
    """
    Legacy adapter kept for imports.

    The MIA pipeline no longer consumes utility metrics or externally computed
    errors. New code should pass PredictionResult directly to run_attacks().
    """

    if model_output is None:
        raise ValueError("model_output deve ser informado.")

    return {
        "prediction_result": model_output,
    }
