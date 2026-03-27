import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _to_numpy(y):
    return y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)


# =========================
# FEATURE ENGINEERING (FORTE)
# =========================
def _attack_features(model, X, y, ref_abs_error=None):
    y_pred = model.predict(X)
    y_true = _to_numpy(y)

    abs_error = np.abs(y_true - y_pred)
    squared_error = (y_true - y_pred) ** 2
    log_abs_error = np.log1p(abs_error)

    # rank (sinal forte)
    if ref_abs_error is not None:
        ref_sorted = np.sort(ref_abs_error)
        rank_error = np.searchsorted(ref_sorted, abs_error) / len(ref_sorted)
    else:
        rank_error = np.zeros_like(abs_error)

    # erro relativo ao comportamento global
    mean_err = abs_error.mean() + 1e-8
    rel_error = abs_error / mean_err

    return np.column_stack([
        abs_error,
        squared_error,
        log_abs_error,
        y_pred,
        rank_error,
        rel_error
    ])


# =========================
# THRESHOLD (SIMPLES E EFICIENTE)
# =========================
def _best_threshold(y_true, y_score):
    thresholds = np.quantile(y_score, np.linspace(0.05, 0.95, 20))

    best_thr = 0.5
    best_score = -1

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        tpr = tp / (tp + fn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)

        score = 0.5 * (tpr + tnr)

        if score > best_score:
            best_score = score
            best_thr = t

    return best_thr


# =========================
# MIA PRINCIPAL
# =========================
def run_membership_inference_attack(
    df,
    target_model,
    X_target_train,
    y_target_train,
    X_target_test,
    y_target_test,
    build_preprocessor,
    n_shadows=10,   # ↓ reduz tempo sem perder muito
    random_state=42,
    target_col="salario"
):
    # ===== preprocessador
    preprocessor = getattr(target_model, "preprocessor_", None)
    if preprocessor is None:
        preprocessor = build_preprocessor(df)
        preprocessor.fit(df.drop(columns=[target_col]))

    test_ratio = len(X_target_test) / (len(X_target_train) + len(X_target_test))

    shadow_X = []
    shadow_y = []
    ref_errors = []

    # ===== SHADOW MODELS
    for i in range(n_shadows):
        df_s, _ = train_test_split(df, test_size=0.5, random_state=random_state + i)

        X = df_s.drop(columns=[target_col])
        y = df_s[target_col]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state + i
        )

        X_tr = preprocessor.transform(X_tr)
        X_te = preprocessor.transform(X_te)

        model = clone(target_model)
        if "random_state" in model.get_params():
            model.set_params(random_state=random_state + i)

        model.fit(X_tr, y_tr)

        # coleta erros para referência global
        ref_errors.append(np.abs(y_tr - model.predict(X_tr)))

        shadow_X.append(_attack_features(model, X_tr, y_tr))
        shadow_y.append(np.ones(len(y_tr)))

        shadow_X.append(_attack_features(model, X_te, y_te))
        shadow_y.append(np.zeros(len(y_te)))

    X_shadow = np.vstack(shadow_X)
    y_shadow = np.concatenate(shadow_y)

    ref_errors = np.concatenate(ref_errors)

    # ===== TREINO DO ATACANTE
    X_train, X_val, y_train, y_val = train_test_split(
        X_shadow, y_shadow, test_size=0.2,
        stratify=y_shadow, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    attacker = LogisticRegression(max_iter=500)
    attacker.fit(X_train, y_train)

    y_val_score = attacker.predict_proba(X_val)[:, 1]
    threshold = _best_threshold(y_val, y_val_score)

    # ===== TARGET
    X_tr = X_target_train
    X_te = X_target_test

    X_attack = np.vstack([
        _attack_features(target_model, X_tr, y_target_train, ref_errors),
        _attack_features(target_model, X_te, y_target_test, ref_errors)
    ])

    y_attack = np.concatenate([
        np.ones(len(X_tr)),
        np.zeros(len(X_te))
    ])

    X_attack = scaler.transform(X_attack)

    y_score = attacker.predict_proba(X_attack)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "y_true": y_attack,
        "y_pred": y_pred,
        "y_score": y_score
    }