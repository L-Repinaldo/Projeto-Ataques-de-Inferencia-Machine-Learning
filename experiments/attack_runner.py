from metrics import compute_attack_metrics
from attacks import run_membership_inference_attack


def run_attacks(df, prediction_result, *, random_state=42):
    """
    Executes the privacy attack without consuming utility metrics.
    """

    mia_output = run_membership_inference_attack(
        df=df,
        target_prediction_result=prediction_result,
        random_state=random_state,
    )

    return compute_attack_metrics(
        y_true=mia_output["y_true"],
        y_pred=mia_output["y_pred"],
    )
