import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_membership_inference_attack(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
):
    """
    Membership Inference Attack baseada na diferença de generalização do modelo alvo.
    """

    eps = 1e-8

    # ERRO RELATIVO → evita vazamento de escala
    train_errors = (np.abs(y_train_true - y_train_pred) / (np.abs(y_train_true) + eps)).to_numpy()
    test_errors  = (np.abs(y_test_true  - y_test_pred)  / (np.abs(y_test_true)  + eps)).to_numpy()

    X_attack = np.concatenate([
        train_errors.reshape(-1,1),
        test_errors.reshape(-1,1)
    ])

    y_attack = np.concatenate([
        np.ones(len(train_errors)),   # membros
        np.zeros(len(test_errors))    # não-membros
    ])

    # Padronização
    scaler = StandardScaler()
    X_attack = scaler.fit_transform(X_attack)

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=42, stratify=y_attack
    )

    attacker = LogisticRegression()
    attacker.fit(X_a_train, y_a_train)

    y_prob = attacker.predict_proba(X_a_test)[:,1]
    y_pred = attacker.predict(X_a_test)

    auc = roc_auc_score(y_a_test, y_prob)
    acc = accuracy_score(y_a_test, y_pred)

    return {
        "attack_auc": float(auc),             
        "attack_accuracy": float(acc),
        "mean_train_error": float(np.mean(train_errors)),
        "mean_test_error": float(np.mean(test_errors)),
        "error_gap": float(np.mean(test_errors) - np.mean(train_errors))
    }
