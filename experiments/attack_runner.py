from metrics import compute_attack_metrics
from attacks import run_membership_inference_attack
from preprocessing import build_preprocessor


def run_attacks(df, target):
    """
    Protocolo experimental padrão do projeto.

    Este método:
    - executa ataque de inferência
    - mede se o índice de vazamentos
    - organiza os resultados

    NÃO:
    - altera datasets
    - aplica DP
    - otimiza modelos
    """


    target_values = target['results']

    
    mia_results = run_membership_inference_attack(
    df = df, target_model= target['results']['model'], X_target_train= target_values["X_train"],
    y_target_train= target_values["y_train_true"], X_target_test= target_values["X_test"],
    y_target_test= target_values["y_test_true"],  build_preprocessor= build_preprocessor
    )


    return {
        "model_name": target['model_name'],
        "results": mia_results,
    }


def attack_metrics(mia_results, model_name):

    results = mia_results['results']

    attack_metrics = compute_attack_metrics(y_true= results['y_true'], y_pred= results['y_pred'])

    return {
        "model_name": model_name,
        "results": attack_metrics
    }
