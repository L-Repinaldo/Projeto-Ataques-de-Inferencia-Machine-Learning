from .impact_classifier import classify_utility_impact, classify_security_risk


def build_summary_table(results):
    table = []
    keys = ["baseline", "eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    model_results = results['Metrics Model']
    attack_results = results['Metrics Attack']

    baseline_mae = model_results["baseline"]['results']["mae"]

    #TODO corrigir esse loop
    for k in keys:
        if not k.startswith("eps_"):
            continue

        mae = model_results[k]["results"]["mae"]
        auc = attack_results[k]["results"]["attack_roc_auc"]

        mae_cv = abs(mae - baseline_mae) / baseline_mae

        row = {
            "modelo": model_results['baseline']['model_name'],
            "epsilon": float(k.split("_")[1]),
            "usabilidade": classify_utility_impact(mae_cv),
            "seguranca": classify_security_risk(auc),
        }

        table.append(row)

    return table
