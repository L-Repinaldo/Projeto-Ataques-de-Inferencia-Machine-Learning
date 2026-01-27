def classify_utility_impact(mae_cv):
    if mae_cv < 0.10:
        return "pouco afetada"
    elif mae_cv < 0.30:
        return "moderadamente afetada"
    else:
        return "muito afetada"


def classify_security_risk(auc):
    if auc < 0.55:
        return "baixo risco"
    elif auc < 0.70:
        return "risco moderado"
    else:
        return "alto risco"
