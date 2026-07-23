import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import build_preprocessor


ATTACK_FEATURE_NAMES = (
    "predicted_value",
    "absolute_error",
    "squared_error",
    "log_absolute_error",
    "relative_error",
    "signed_error",
    "error_rank",
    "prediction_rank",
)


def run_membership_inference_attack(
    df,
    target_prediction_result,
    *,
    target_col="salario",
    n_shadows=10,
    shadow_dataset_fraction=0.8,
    balance_attack_training=True,
    balance_target_evaluation=True,
    random_state=42,
):
    """
    Shadow Model Membership Inference Attack for regression models.

    The attacker is trained only on outputs produced by shadow models. Utility
    metrics are not consumed by this implementation.
    """

    target_model = target_prediction_result.model
    shadow_test_size = _resolve_test_size(target_prediction_result)

    shadow_features = []
    shadow_membership = []

    for shadow_id in range(n_shadows):
        shadow_df = _build_shadow_dataset(
            df=df,
            target_col=target_col,
            fraction=shadow_dataset_fraction,
            random_state=random_state + shadow_id,
        )

        X_shadow, y_shadow = _train_shadow_and_collect_outputs(
            df_shadow=shadow_df,
            target_model=target_model,
            target_col=target_col,
            test_size=shadow_test_size,
            random_state=random_state + shadow_id,
        )

        shadow_features.append(X_shadow)
        shadow_membership.append(y_shadow)

    X_attack = np.vstack(shadow_features)
    y_attack = np.concatenate(shadow_membership)

    if balance_attack_training:
        attack_indices = _balanced_indices(
            y=y_attack,
            random_state=random_state,
        )
        X_attack = X_attack[attack_indices]
        y_attack = y_attack[attack_indices]

    scaler = StandardScaler()
    X_attack = scaler.fit_transform(X_attack)

    attacker = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    attacker.fit(X_attack, y_attack)

    X_eval, y_true = _target_evaluation_dataset(
        prediction_result=target_prediction_result,
        random_state=random_state,
        balance=balance_target_evaluation,
    )

    X_eval = scaler.transform(X_eval)
    y_pred = attacker.predict(X_eval)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "feature_names": ATTACK_FEATURE_NAMES,
    }


def _resolve_test_size(prediction_result):
    if prediction_result.test_size is None:
        return 0.3

    return prediction_result.test_size


def _build_shadow_dataset(df, *, target_col, fraction, random_state):
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' nao encontrado no dataset.")

    if not 0 < fraction <= 1:
        raise ValueError("shadow_dataset_fraction deve estar no intervalo (0, 1].")

    if fraction == 1:
        return df.sample(
            frac=1,
            replace=True,
            random_state=random_state,
        ).reset_index(drop=True)

    shadow_df, _ = train_test_split(
        df,
        train_size=fraction,
        random_state=random_state,
    )

    return shadow_df.reset_index(drop=True)


def _train_shadow_and_collect_outputs(
    *,
    df_shadow,
    target_model,
    target_col,
    test_size,
    random_state,
):
    X = df_shadow.drop(columns=[target_col])
    y = df_shadow[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(df=df_shadow)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    shadow_model = _clone_shadow_model(
        target_model=target_model,
        random_state=random_state,
    )

    shadow_model.fit(X_train_enc, y_train)

    train_features = _build_attack_features(
        y_true=y_train,
        y_pred=shadow_model.predict(X_train_enc),
    )
    test_features = _build_attack_features(
        y_true=y_test,
        y_pred=shadow_model.predict(X_test_enc),
    )

    X_attack = np.vstack([train_features, test_features])
    y_attack = np.concatenate([
        np.ones(len(train_features)),
        np.zeros(len(test_features)),
    ])

    return X_attack, y_attack


def _clone_shadow_model(*, target_model, random_state):
    shadow_model = clone(target_model)

    if "random_state" in shadow_model.get_params():
        shadow_model.set_params(random_state=random_state)

    return shadow_model


def _target_evaluation_dataset(*, prediction_result, random_state, balance):
    member_features = _build_attack_features(
        y_true=prediction_result.y_train_true,
        y_pred=prediction_result.y_train_pred,
    )
    non_member_features = _build_attack_features(
        y_true=prediction_result.y_test_true,
        y_pred=prediction_result.y_test_pred,
    )

    X_eval = np.vstack([member_features, non_member_features])
    y_true = np.concatenate([
        np.ones(len(member_features)),
        np.zeros(len(non_member_features)),
    ])

    if balance:
        eval_indices = _balanced_indices(
            y=y_true,
            random_state=random_state,
        )
        X_eval = X_eval[eval_indices]
        y_true = y_true[eval_indices]

    return X_eval, y_true


def _build_attack_features(*, y_true, y_pred):
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)

    signed_error = y_pred - y_true
    absolute_error = np.abs(signed_error)
    squared_error = signed_error ** 2
    log_absolute_error = np.log1p(absolute_error)
    relative_error = absolute_error / (np.abs(y_true) + 1e-8)
    error_rank = _percent_rank(absolute_error)
    prediction_rank = _percent_rank(y_pred)

    return np.column_stack([
        y_pred,
        absolute_error,
        squared_error,
        log_absolute_error,
        relative_error,
        signed_error,
        error_rank,
        prediction_rank,
    ])


def _as_float_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _percent_rank(values):
    values = _as_float_array(values)

    if len(values) == 0:
        return values

    if len(values) == 1:
        return np.array([0.5])

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values)) / (len(values) - 1)

    return ranks


def _balanced_indices(*, y, random_state):
    y = np.asarray(y)

    member_indices = np.flatnonzero(y == 1)
    non_member_indices = np.flatnonzero(y == 0)

    sample_size = min(len(member_indices), len(non_member_indices))

    if sample_size == 0:
        return np.arange(len(y))

    rng = np.random.default_rng(random_state)
    member_sample = rng.choice(member_indices, size=sample_size, replace=False)
    non_member_sample = rng.choice(non_member_indices, size=sample_size, replace=False)

    indices = np.concatenate([member_sample, non_member_sample])
    rng.shuffle(indices)

    return indices
