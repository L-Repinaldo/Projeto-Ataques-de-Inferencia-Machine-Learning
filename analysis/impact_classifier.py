def classify_utility_impact(mae_cv):
    if mae_cv < 0.50:
        return "pouco afetada"
    elif mae_cv < 1.50:
        return "moderadamente afetada"
    else:
        return "muito afetada"


def classify_leakage_risk(adv):
    if adv < 0:
        return "pior que aleatório"
    elif adv < 0.01:
        return "quase aleatório"
    elif adv < 0.05:
        return "vazamento muito fraco"
    elif adv < 0.12:
        return "vazamento fraco"
    elif adv < 0.25:
        return "vazamento moderado"
    elif adv < 0.45:
        return "vazamento alto"
    else:
        return "vazamento muito alto"

