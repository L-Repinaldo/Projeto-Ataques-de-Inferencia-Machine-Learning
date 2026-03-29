from statistics import mean
from experiments import (
    run_model, 
    run_attacks,
)

SEEDS = [42, 123, 2026]
TEST_SIZES = [0.2, 0.3]

def _aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()

    results = {}
    for k in keys:
        values = [m[k] for m in metrics_list]

        if isinstance(values[0], (int, float)):
            results[k] = round(mean(values), 3)

    return results

def run_machine_learning_experiments(model_runner, model_name, datasets, dataset_names):

    """
    Protocolo experimental padrão do projeto.

    Este método:
    - Chama os métodos responsáveis pelo experimento da aplicação 
    - organiza os resultados

    NÃO:
    - altera datasets
    - Executa experimento
    """

    model_experiment_output = {}
    attack_experiment_output = {}

    for name, df in zip(dataset_names, datasets):

        model_runs = []
        attack_runs = []

        for seed in SEEDS:
            for test_size in TEST_SIZES:

                model_metrics_values = run_model(
                    df=df, 
                    model_runner=lambda **kwargs: model_runner(
                        **kwargs,
                        seed=seed,
                        test_size=test_size
                    )
                )

                attack_metrics_values = run_attacks(target= model_metrics_values)


                model_runs.append(model_metrics_values)
                attack_runs.append(attack_metrics_values)

        model_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(model_runs),
        }

        attack_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(attack_runs),
        }

    return model_experiment_output, attack_experiment_output