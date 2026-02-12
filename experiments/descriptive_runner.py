
from metrics import compute_descriptive_metrics

def run_descriptive_statistics(name, df, baseline_values):
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

    salario_values = df['salario']

    results = {}
    
    results = compute_descriptive_metrics(info_baseline= baseline_values, df= salario_values)
    
    return results


    