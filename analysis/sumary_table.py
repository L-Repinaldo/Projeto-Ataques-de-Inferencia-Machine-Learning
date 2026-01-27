from .impact_classifier import classify_utility_impact, classify_security_risk


def build_summary_table(results, model_name):
    table = []

    baseline_mae = results["baseline"]["utility"]["mae"]

    for k, v in results.items():
        if not k.startswith("eps_"):
            continue

        mae = v["utility"]["mae"]
        auc = v["mia"]["attack_auc"]

        mae_cv = abs(mae - baseline_mae) / baseline_mae

        row = {
            "modelo": model_name,
            "epsilon": float(k.split("_")[1]),
            "usabilidade": classify_utility_impact(mae_cv),
            "seguranca": classify_security_risk(auc),
        }

        table.append(row)

    return table
