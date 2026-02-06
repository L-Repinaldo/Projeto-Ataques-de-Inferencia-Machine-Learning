from metrics import compute_utility_metrics, compute_attack_metrics
from attacks import run_membership_inference_attack
from preprocessing import build_preprocessor


def run_experiment(model_runner, model_name, datasets, dataset_names):
    """
    Protocolo experimental padrão do projeto.

    Este método:
    - executa o modelo como instrumento de medição
    - mede utilidade
    - mede estabilidade
    - executa ataque de inferência
    - organiza os resultados

    NÃO:
    - altera datasets
    - aplica DP
    - otimiza modelos
    """

    results = {}


    for name, df in zip(dataset_names, datasets):

        preprocessor = build_preprocessor(df=df)

        run_output = model_runner(
            df=df,
            preprocessor=preprocessor,
        )

        utility = compute_utility_metrics(
            y_true=run_output["y_test_true"],
            y_pred=run_output["y_test_pred"],
        )

        results[name] = {
            "utility": utility,
            "n_features": run_output["n_features"],
        }

        mia_results = run_membership_inference_attack(
            df = df,
            target_model=run_output["model"],
            X_target_train=run_output["X_train"],
            y_target_train=run_output["y_train_true"],
            X_target_test=run_output["X_test"],
            y_target_test=run_output["y_test_true"],
            build_preprocessor= build_preprocessor
        )


        results[name]["mia"] = mia_results

        attack_metrics = compute_attack_metrics(
            y_attack_target= mia_results['y_attack_target'],
            y_pred= mia_results['y_predicted'],
            y_pred_proba= mia_results['y_prediction_probability']
        )

        results[name]["mia"]["attack_metrics"] = attack_metrics


    return {
        "model_name": model_name,
        "results": results,
    }
