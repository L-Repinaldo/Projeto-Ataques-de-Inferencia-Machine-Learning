from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score


def compute_attack_metrics(y_attack_target, y_pred, y_pred_proba):
    """
    Calcula métricas básicas da eficacia das classificações do ataque.
    """ 

    conf_matrix = confusion_matrix(y_true= y_attack_target, y_pred= y_pred)

    accuracy = accuracy_score(y_true= y_attack_target, y_pred= y_pred) 

    precision = precision_score(y_true= y_attack_target, y_pred= y_pred)

    f1 = f1_score(y_true= y_attack_target, y_pred= y_pred)
    
    fpr, tpr, tresholds = roc_curve(y_true= y_attack_target, y_score= y_pred_proba)

    roc_auc = roc_auc_score(y_true= y_attack_target, y_score= y_pred_proba)


    return {
        "attack_confusion_matrix" : conf_matrix,
        "attack_accuracy": accuracy,
        "attack_precision": precision,
        "attack_f1_score": f1,
        "attack_roc_curve": [fpr, tpr],
        "attack_roc_auc": roc_auc,
    }