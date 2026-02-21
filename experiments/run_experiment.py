from experiments import (
    run_model, 
    run_attacks,
    model_metrics,
    attack_metrics,
    )


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

        model_results = run_model(name= name, df= df, model_name= model_name, model_runner= model_runner)
        model_metrics_values  = model_metrics(model_output= model_results, dataset_name= name, model_name= model_name)
        model_experiment_output[name] = model_metrics_values

        attack_results = run_attacks(df= df, target= model_results)
        attack_metrics_values = attack_metrics(model_name= model_name, mia_results= attack_results)
        attack_experiment_output[name] = attack_metrics_values



    return model_experiment_output, attack_experiment_output
