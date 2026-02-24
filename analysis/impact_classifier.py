def classify_utility_impact(mae_cv):
    if mae_cv < 0.50:
        return "pouco afetada"
    elif mae_cv < 1.50:
        return "moderadamente afetada"
    else:
        return "muito afetada"


def classify_security_risk(accuracy):
    if accuracy < 0.55:
        return "quase aleatório"
    elif accuracy < 0.62:
        return "vazamento fraco"
    elif accuracy < 0.70:
        return "vazamento moderado"
    else:
        return "vazamento forte"

