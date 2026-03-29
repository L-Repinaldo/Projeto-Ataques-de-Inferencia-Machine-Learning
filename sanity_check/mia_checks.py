from __future__ import annotations

import numpy as np

from attacks import run_membership_inference_attack
from metrics import compute_attack_metrics

from .common import DEFAULT_SEED


def _status_by_range(value: float, low: float, high: float) -> str:
    return "ok" if low <= value <= high else "suspeito"


def _run_attack_metrics(target_outputs):
    mia_output = run_membership_inference_attack(target_outputs=target_outputs)
    return compute_attack_metrics(
        y_true=mia_output["y_test"],
        y_pred=mia_output["y_pred"],
    )


def _as_array(x):
    return np.asarray(x)


def test_random_label(target_outputs, *, seed: int = DEFAULT_SEED):
    """
    Shuffle errors so train/test distributions match. Attack should be near random.
    """
    rng = np.random.default_rng(seed)
    train_err = _as_array(target_outputs["train_abs_error"])
    test_err = _as_array(target_outputs["test_abs_error"])

    combined = np.concatenate([train_err, test_err])
    rng.shuffle(combined)

    train_size = len(train_err)
    new_train = combined[:train_size]
    new_test = combined[train_size:]

    metrics = _run_attack_metrics(
        {"train_abs_error": new_train, "test_abs_error": new_test}
    )
    status = _status_by_range(metrics["attack_acc"], 0.45, 0.55)
    return {
        "test": "random_label",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
    }


def test_train_vs_train(target_outputs):
    """
    Use the same errors for members and non-members. Expect near random.
    """
    train_err = _as_array(target_outputs["train_abs_error"])
    metrics = _run_attack_metrics(
        {"train_abs_error": train_err, "test_abs_error": train_err.copy()}
    )
    status = _status_by_range(metrics["attack_acc"], 0.45, 0.55)
    return {
        "test": "train_vs_train",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
    }


def test_train_vs_noise(target_outputs, *, seed: int = DEFAULT_SEED):
    """
    Use high-noise errors as non-members. Expect high attack accuracy.
    """
    rng = np.random.default_rng(seed)
    train_err = _as_array(target_outputs["train_abs_error"])
    test_err = _as_array(target_outputs["test_abs_error"])

    base_mean = float(np.mean(train_err))
    base_std = float(np.std(train_err)) + 1e-9

    noise = np.abs(rng.normal(loc=0.0, scale=base_std * 2.0, size=len(test_err)))
    noisy_test = test_err + noise + base_mean

    metrics = _run_attack_metrics(
        {"train_abs_error": train_err, "test_abs_error": noisy_test}
    )
    status = "ok" if metrics["attack_acc"] >= 0.7 else "suspeito"
    return {
        "test": "train_vs_noise",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
    }


def test_overfitting_amplificado(target_outputs):
    """
    Amplify the train/test error gap. Expect higher accuracy than baseline.
    """
    train_err = _as_array(target_outputs["train_abs_error"])
    test_err = _as_array(target_outputs["test_abs_error"])

    baseline = _run_attack_metrics(target_outputs)["attack_acc"]

    amplified_train = train_err * 0.5
    amplified_test = test_err * 1.5 + np.mean(train_err)

    metrics = _run_attack_metrics(
        {"train_abs_error": amplified_train, "test_abs_error": amplified_test}
    )
    status = "ok" if metrics["attack_acc"] >= baseline + 0.05 else "suspeito"
    return {
        "test": "overfitting_amplificado",
        "attack_acc": metrics["attack_acc"],
        "baseline_acc": baseline,
        "status": status,
    }


def test_balanceamento_dataset_ataque(target_outputs):
    """
    Basic check: ensure both member and non-member sets are non-empty.
    """
    train_err = _as_array(target_outputs["train_abs_error"])
    test_err = _as_array(target_outputs["test_abs_error"])

    total = len(train_err) + len(test_err)
    ratio = len(train_err) / max(total, 1)

    status = "ok" if 0.1 <= ratio <= 0.9 else "suspeito"
    return {
        "test": "balanceamento_dataset_ataque",
        "target_member_ratio": ratio,
        "status": status,
    }


def run_all_mia_sanity_checks(target_outputs):
    results = []
    results.append(test_random_label(target_outputs))
    results.append(test_train_vs_train(target_outputs))
    results.append(test_train_vs_noise(target_outputs))
    results.append(test_overfitting_amplificado(target_outputs))
    results.append(test_balanceamento_dataset_ataque(target_outputs))
    return results
