from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from metrics import compute_utility_metrics

from .common import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    TARGET_COL,
    run_model_and_collect,
    split_dataset,
)


def _status_by_expected(condition_ok: bool) -> str:
    return "ok" if condition_ok else "suspeito"


def _safe_set_params(model, **kwargs):
    params = model.get_params()
    allowed = {k: v for k, v in kwargs.items() if k in params}
    if allowed:
        model.set_params(**allowed)


def _mean_abs_error(errors):
    return float(np.mean(np.asarray(errors)))


def test_random_label(base_outputs, model_name, *, seed: int = DEFAULT_SEED):
    """
    Random label test: shuffle labels in training, expect worse performance.
    """
    rng = np.random.default_rng(seed)

    model = clone(base_outputs.model)
    y_train_shuffled = rng.permutation(base_outputs.y_train_true)

    model.fit(base_outputs.X_train, y_train_shuffled)
    y_train_pred = model.predict(base_outputs.X_train)
    y_test_pred = model.predict(base_outputs.X_test)

    metrics = compute_utility_metrics(
        y_train_true=base_outputs.y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=base_outputs.y_test_true,
        y_test_pred=y_test_pred,
    )

    base_mae = _mean_abs_error(base_outputs.utility["test_abs_error"])
    mae = metrics["mae"]
    status = _status_by_expected(mae > base_mae * 1.1)
    return {
        "model": model_name,
        "test": "random_label",
        "mae": mae,
        "baseline_mae": base_mae,
        "status": status,
    }


def test_overfitting_controlado(base_outputs, model_name):
    """
    Overfitting test: increase capacity and reduce data.
    """
    model = clone(base_outputs.model)
    _safe_set_params(
        model,
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=1,
        min_samples_split=2,
        min_child_weight=1,
    )

    model.fit(base_outputs.X_train, base_outputs.y_train_true)
    y_train_pred = model.predict(base_outputs.X_train)
    y_test_pred = model.predict(base_outputs.X_test)

    metrics = compute_utility_metrics(
        y_train_true=base_outputs.y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=base_outputs.y_test_true,
        y_test_pred=y_test_pred,
    )
    train_mae = _mean_abs_error(metrics["train_abs_error"])
    test_mae = _mean_abs_error(metrics["test_abs_error"])
    gap = test_mae - train_mae

    status = _status_by_expected(gap > 0.1 * max(train_mae, 1e-9))
    return {
        "model": model_name,
        "test": "overfitting_controlado",
        "train_mae": train_mae,
        "test_mae": test_mae,
        "mae_gap": gap,
        "status": status,
    }


def test_underfitting_controlado(base_outputs, model_name):
    """
    Underfitting test: reduce capacity and expect higher train error.
    """
    model = clone(base_outputs.model)
    _safe_set_params(
        model,
        n_estimators=50,
        max_depth=2,
        min_samples_leaf=30,
        min_samples_split=60,
        min_child_weight=50,
    )

    model.fit(base_outputs.X_train, base_outputs.y_train_true)
    y_train_pred = model.predict(base_outputs.X_train)
    y_test_pred = model.predict(base_outputs.X_test)

    metrics = compute_utility_metrics(
        y_train_true=base_outputs.y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=base_outputs.y_test_true,
        y_test_pred=y_test_pred,
    )
    train_mae = _mean_abs_error(metrics["train_abs_error"])
    test_mae = _mean_abs_error(metrics["test_abs_error"])
    gap = test_mae - train_mae

    status = _status_by_expected(train_mae > 0.9 * test_mae)
    return {
        "model": model_name,
        "test": "underfitting_controlado",
        "train_mae": train_mae,
        "test_mae": test_mae,
        "mae_gap": gap,
        "status": status,
    }


def test_data_leakage(base_outputs, model_name):
    """
    Data leakage check: train error should be lower than test error.
    """
    train_mae = _mean_abs_error(base_outputs.utility["train_abs_error"])
    test_mae = _mean_abs_error(base_outputs.utility["test_abs_error"])

    status = _status_by_expected(test_mae > train_mae)
    return {
        "model": model_name,
        "test": "data_leakage",
        "train_mae": train_mae,
        "test_mae": test_mae,
        "status": status,
    }


def test_stability_across_seeds(df_baseline, model_runner, model_name):
    """
    Stability across seeds: low variance in test MAE.
    """
    seeds = [42, 123, 2026, 77, 99]
    maes = []

    for seed in seeds:
        outputs = run_model_and_collect(
            df_baseline,
            model_runner,
            seed=seed,
            test_size=DEFAULT_TEST_SIZE,
            target=TARGET_COL,
        )
        maes.append(_mean_abs_error(outputs.utility["test_abs_error"]))

    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes))
    status = _status_by_expected(std_mae / (mean_mae + 1e-9) < 0.1)
    return {
        "model": model_name,
        "test": "stability_across_seeds",
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "status": status,
    }


def test_sensibilidade_tamanho_dataset(df_baseline, base_outputs, model_name):
    """
    Sensitivity to dataset size: MAE should improve with more data.
    """
    X_train_raw, X_test_raw, y_train_full, y_test_full = split_dataset(
        df_baseline, target=TARGET_COL, test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED
    )

    preprocessor = base_outputs.preprocessor
    X_test = preprocessor.transform(X_test_raw)

    fracs = [0.3, 0.6, 1.0]
    results = []

    for frac in fracs:
        if frac >= 1.0:
            X_train_part = X_train_raw
            y_train_part = y_train_full
        else:
            X_train_part, _, y_train_part, _ = train_test_split(
                X_train_raw,
                y_train_full,
                train_size=frac,
                random_state=DEFAULT_SEED,
            )

        X_train = preprocessor.transform(X_train_part)
        model = clone(base_outputs.model)
        _safe_set_params(model, random_state=DEFAULT_SEED)
        model.fit(X_train, y_train_part)

        y_test_pred = model.predict(X_test)
        metrics = compute_utility_metrics(
            y_train_true=y_train_part,
            y_train_pred=model.predict(X_train),
            y_test_true=y_test_full,
            y_test_pred=y_test_pred,
        )
        mae = metrics["mae"]
        results.append((frac, mae))

    status = _status_by_expected(
        results[0][1] >= results[1][1] * 0.98 and results[1][1] >= results[2][1] * 0.98
    )
    return {
        "model": model_name,
        "test": "sensibilidade_tamanho_dataset",
        "mae_30": results[0][1],
        "mae_60": results[1][1],
        "mae_100": results[2][1],
        "status": status,
    }


def test_feature_importance(base_outputs, model_name):
    """
    Feature importance check: avoid extreme concentration.
    """
    model = base_outputs.model
    preprocessor = base_outputs.preprocessor

    feature_names = None
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_booster"):
        score = model.get_booster().get_score(importance_type="gain")
        if not score:
            return {
                "model": model_name,
                "test": "feature_importance",
                "status": "suspeito",
                "reason": "sem_importancia",
            }
        importances = np.asarray(list(score.values()), dtype=float)
    else:
        return {
            "model": model_name,
            "test": "feature_importance",
            "status": "suspeito",
            "reason": "indisponivel",
        }

    total = float(np.sum(importances)) + 1e-9
    shares = importances / total
    max_share = float(np.max(shares))
    top3_share = float(np.sort(shares)[-3:].sum())

    top_features = []
    top_idx = np.argsort(shares)[-5:][::-1]
    if feature_names and len(feature_names) == len(importances):
        top_features = [
            {"feature": feature_names[i], "share": float(shares[i])}
            for i in top_idx
        ]
    else:
        top_features = [
            {"feature": f"f{i}", "share": float(shares[i])}
            for i in top_idx
        ]

    status = _status_by_expected(max_share < 0.85 and top3_share < 0.95)
    return {
        "model": model_name,
        "test": "feature_importance",
        "max_importance_share": max_share,
        "top3_importance_share": top3_share,
        "status": status,
        "top_features": top_features,
    }


def run_all_model_sanity_checks(df_baseline, model_runner, model_name):
    base_outputs = run_model_and_collect(
        df_baseline,
        model_runner,
        seed=DEFAULT_SEED,
        test_size=DEFAULT_TEST_SIZE,
        target=TARGET_COL,
    )

    results = []
    results.append(test_random_label(base_outputs, model_name, seed=DEFAULT_SEED))
    results.append(test_overfitting_controlado(base_outputs, model_name))
    results.append(test_underfitting_controlado(base_outputs, model_name))
    results.append(test_data_leakage(base_outputs, model_name))
    results.append(test_stability_across_seeds(df_baseline, model_runner, model_name))
    results.append(test_sensibilidade_tamanho_dataset(df_baseline, base_outputs, model_name))
    results.append(test_feature_importance(base_outputs, model_name))
    return results
