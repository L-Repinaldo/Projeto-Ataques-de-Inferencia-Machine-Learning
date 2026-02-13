def classify_utility_impact(mae_cv):
    if mae_cv < 0.50:
        return "pouco afetada"
    elif mae_cv < 1.50:
        return "moderadamente afetada"
    else:
        return "muito afetada"


def classify_security_risk(auc):
    if auc < 0.52:
        return "quase aleatório"
    elif auc < 0.60:
        return "vazamento fraco"
    elif auc < 0.75:
        return "vazamento moderado"
    else:
        return "vazamento forte"

