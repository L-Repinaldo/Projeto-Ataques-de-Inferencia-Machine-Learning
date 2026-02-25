import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


def extract_attack_features(model, X, y):
    y_pred = model.predict(X)
    loss = np.abs(y.to_numpy() - y_pred)
    squared_loss = (y.to_numpy() - y_pred) ** 2

    if hasattr(model, "estimators_"):  # Random Forest
        preds = np.stack([tree.predict(X) for tree in model.estimators_])
        pred_std = preds.std(axis=0)

    elif hasattr(model, "get_booster"):  # XGBoost
        noise = np.random.normal(0, 1e-6, size=X.shape)
        pred_std = np.abs(model.predict(X + noise) - y_pred)
    else:
        pred_std = np.zeros_like(loss)
    
    return np.column_stack([loss, squared_loss, pred_std])

def run_membership_inference_attack(
    df, target_model, X_target_train,
    y_target_train, X_target_test, y_target_test,
    build_preprocessor, n_shadows=3, random_state=42
):

    X_attack_shadow_all = []
    y_attack_shadow_all = []

    for i in range(n_shadows):
        shadow_df, _ = train_test_split(
            df, test_size=0.5, random_state=random_state + i
        )

        X_shadow_raw = shadow_df.drop(columns=["salario"])
        y_shadow = shadow_df["salario"]

        shadow_preprocessor = build_preprocessor(df=shadow_df)
        X_shadow = shadow_preprocessor.fit_transform(X_shadow_raw)

        X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
            X_shadow, y_shadow, test_size=0.5, random_state=random_state + i
        )

        shadow_model = clone(target_model)
        if "random_state" in shadow_model.get_params():
            shadow_model.set_params(random_state=random_state + i)

        shadow_model.fit(X_s_train, y_s_train)

        X_attack_shadow_train = extract_attack_features(shadow_model, X_s_train, y_s_train)
        X_attack_shadow_test  = extract_attack_features(shadow_model, X_s_test,  y_s_test)

        X_attack_shadow_all.append(
            np.vstack([X_attack_shadow_train, X_attack_shadow_test])
        )
        y_attack_shadow_all.append(
            np.concatenate([
                np.ones(len(X_attack_shadow_train)),
                np.zeros(len(X_attack_shadow_test))
            ])
        )

    X_attack_shadow = np.vstack(X_attack_shadow_all)
    y_attack_shadow = np.concatenate(y_attack_shadow_all)

    scaler = StandardScaler()
    X_attack_shadow = scaler.fit_transform(X_attack_shadow)

    attacker = LogisticRegression(max_iter=1000)
    attacker.fit(X_attack_shadow, y_attack_shadow)

    X_attack_target_train = extract_attack_features(target_model, X_target_train, y_target_train)
    X_attack_target_test  = extract_attack_features(target_model, X_target_test,  y_target_test)

    X_attack_target = np.vstack([X_attack_target_train, X_attack_target_test])
    y_attack_target = np.concatenate([
        np.ones(len(X_attack_target_train)),
        np.zeros(len(X_attack_target_test))
    ])

    X_attack_target = scaler.transform(X_attack_target)

    y_pred_proba = attacker.predict_proba(X_attack_target)[:, 1]
    y_pred = attacker.predict(X_attack_target)

    return {
        "y_attack_target" : y_attack_target,
        "y_predicted": y_pred,
        "y_prediction_probability": y_pred_proba,
    }
