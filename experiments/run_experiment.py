from statistics import mean
from experiments import (
    run_model, 
    run_attacks,
    model_metrics,
    attack_metrics,
)

SEEDS = [42, 123, 2026]
TEST_SIZES = [0.2, 0.3]

def _round_dict(d, ndigits=3):
    return {k: round(v, ndigits) for k, v in d.items()}

def _aggregate_metrics(metrics_list):
    return {
        k: round(mean(m[k] for m in metrics_list), 3)
        for k in metrics_list[0].keys()
    }

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

                model_results = run_model(name=name, df=df, model_name=model_name,
                    model_runner=lambda **kwargs: model_runner(
                        **kwargs,
                        seed=seed,
                        test_size=test_size
                    )
                )

                model_metrics_values = model_metrics(
                    model_output=model_results,
                    dataset_name=name,
                    model_name=model_name
                )

                attack_results = run_attacks(df= df, target= model_results)

                attack_metrics_values = attack_metrics(
                    model_name=model_name,
                    mia_results=attack_results
                )

                model_runs.append(model_metrics_values["results"])
                attack_runs.append(attack_metrics_values["results"])

        model_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(model_runs),
        }

        attack_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(attack_runs),
        }

    return model_experiment_output, attack_experiment_output