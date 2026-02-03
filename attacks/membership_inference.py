import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def run_membership_inference_attack(
    df,
    target_model,
    X_target_train,
    y_target_train,
    X_target_test,
    y_target_test,
    build_preprocessor,
):

    # ================= SHADOW DATA =================
    shadow_df, _ = train_test_split(df, test_size=0.5, random_state=42)

    X_shadow_raw = shadow_df.drop(columns=["salario"])
    y_shadow = shadow_df["salario"]

    shadow_preprocessor = build_preprocessor(df=shadow_df)
    X_shadow = shadow_preprocessor.fit_transform(X_shadow_raw)

    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_shadow, y_shadow, test_size=0.5, random_state=42
    )

    # CLONA MODELO ALVO
    shadow_model = clone(target_model)
    shadow_model.fit(X_s_train, y_s_train)

    # LOSS
    shadow_train_loss = np.abs(y_s_train.to_numpy() - shadow_model.predict(X_s_train))
    shadow_test_loss  = np.abs(y_s_test.to_numpy()  - shadow_model.predict(X_s_test))


    X_attack_shadow = np.concatenate([
        shadow_train_loss.reshape(-1,1),
        shadow_test_loss.reshape(-1,1)
    ])
    y_attack_shadow = np.concatenate([
        np.ones(len(shadow_train_loss)),
        np.zeros(len(shadow_test_loss))
    ])

    scaler = StandardScaler()
    X_attack_shadow = scaler.fit_transform(X_attack_shadow)

    attacker = LogisticRegression()
    attacker.fit(X_attack_shadow, y_attack_shadow)

    # ================= TARGET =================
    target_train_loss = np.abs(y_target_train.to_numpy() - target_model.predict(X_target_train))
    target_test_loss  = np.abs(y_target_test.to_numpy()  - target_model.predict(X_target_test))

    X_attack_target = np.concatenate([
        target_train_loss.reshape(-1,1),
        target_test_loss.reshape(-1,1)
    ])
    y_attack_target = np.concatenate([
        np.ones(len(target_train_loss)),
        np.zeros(len(target_test_loss))
    ])

    X_attack_target = scaler.transform(X_attack_target)

    y_prob = attacker.predict_proba(X_attack_target)[:,1]
    y_pred = attacker.predict(X_attack_target)

    return {
        "attack_auc": float(roc_auc_score(y_attack_target, y_prob)),
        "attack_accuracy": float(accuracy_score(y_attack_target, y_pred)),
        "mean_train_loss": float(np.mean(target_train_loss)),
        "mean_test_loss": float(np.mean(target_test_loss)),
        "generalization_gap": float(np.mean(target_test_loss) - np.mean(target_train_loss))
    }
