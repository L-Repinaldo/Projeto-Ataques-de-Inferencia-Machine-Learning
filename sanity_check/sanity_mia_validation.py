from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pprint
import warnings

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from data import load_data
from experiments import run_model
from attacks import run_membership_inference_attack
from metrics import compute_attack_metrics
from preprocessing import build_preprocessor
from model import run_random_forest, run_xgboost, run_gradient_boosting, run_extra_trees

import attacks.membership_inference as mia


SEED = 42
TEST_SIZE = 0.3


def _status_by_expected(attack_acc, expected_low, expected_high):
    if expected_low <= attack_acc <= expected_high:
        return "ok"
    return "suspeito"


def _get_target_test_size(X_train, X_test):
    return len(X_test) / (len(X_train) + len(X_test))


def _extract_features(model, X, y):
    return mia._extract_features_blackbox_array(model, X, y)


def _compute_feature_stats(X):
    # Estatisticas agregadas para comparar distribuicao de features
    return {
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
    }


def test_random_label(df_baseline, model_runner, model_name):
    """
    TESTE 1 - RANDOM LABEL TEST
    Valida se o MIA colapsa para aleatorio quando o modelo e treinado com rotulos embaralhados.
    """
    rng = np.random.default_rng(SEED)

    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    target_model = clone(base_results["results"]["model"])
    X_train = base_results["results"]["X_train"]
    X_test = base_results["results"]["X_test"]
    y_train = base_results["results"]["y_train_true"]

    y_train_shuffled = rng.permutation(y_train)
    target_model.fit(X_train, y_train_shuffled)
    target_model.preprocessor_ = base_results["results"]["model"].preprocessor_

    mia_results = run_membership_inference_attack(
        df=df_baseline,
        target_model=target_model,
        X_target_train=X_train,
        y_target_train=base_results["results"]["y_train_true"],
        X_target_test=X_test,
        y_target_test=base_results["results"]["y_test_true"],
        build_preprocessor=build_preprocessor,
    )

    metrics = compute_attack_metrics(
        y_true=mia_results["y_true"],
        y_pred=mia_results["y_pred"]
    )

    status = _status_by_expected(metrics["attack_acc"], 0.48, 0.52)
    return {
        "test": "random_label",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
        "debug": mia_results.get("debug"),
    }


def test_train_vs_train(df_baseline, model_runner, model_name):
    """
    TESTE 2 - TRAIN VS TRAIN
    Usa o mesmo conjunto como membros e nao-membros.
    """
    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    X_train = base_results["results"]["X_train"]
    y_train = base_results["results"]["y_train_true"]

    mia_results = run_membership_inference_attack(
        df=df_baseline,
        target_model=base_results["results"]["model"],
        X_target_train=X_train,
        y_target_train=y_train,
        X_target_test=X_train,
        y_target_test=y_train,
        build_preprocessor=build_preprocessor,
    )

    metrics = compute_attack_metrics(
        y_true=mia_results["y_true"],
        y_pred=mia_results["y_pred"]
    )

    status = _status_by_expected(metrics["attack_acc"], 0.48, 0.52)
    return {
        "test": "train_vs_train",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
        "debug": mia_results.get("debug"),
    }


def test_train_vs_noise(df_baseline, model_runner, model_name):
    """
    TESTE 3 - TRAIN VS NOISE
    Usa ruido como nao-membros para verificar se o ataque detecta diferenca extrema.
    """
    rng = np.random.default_rng(SEED)

    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    X_train = base_results["results"]["X_train"]
    y_train = base_results["results"]["y_train_true"]
    X_test = base_results["results"]["X_test"]

    noise = rng.normal(loc=0.0, scale=1.0, size=X_test.shape)
    y_noise = rng.choice(np.asarray(y_train), size=noise.shape[0], replace=True)

    mia_results = run_membership_inference_attack(
        df=df_baseline,
        target_model=base_results["results"]["model"],
        X_target_train=X_train,
        y_target_train=y_train,
        X_target_test=noise,
        y_target_test=y_noise,
        build_preprocessor=build_preprocessor,
    )

    metrics = compute_attack_metrics(
        y_true=mia_results["y_true"],
        y_pred=mia_results["y_pred"]
    )

    status = "ok" if metrics["attack_acc"] >= 0.7 else "suspeito"
    return {
        "test": "train_vs_noise",
        "attack_acc": metrics["attack_acc"],
        "member_acc": metrics["member_acc"],
        "non_member_acc": metrics["non_member_acc"],
        "status": status,
        "debug": mia_results.get("debug"),
    }


def test_overfitting_amplificado(df_baseline, model_runner, model_name):
    """
    TESTE 4 - OVERFITTING AMPLIFICADO
    Reduz dataset e aumenta capacidade do modelo para verificar aumento de vazamento.
    """
    df_small = df_baseline.sample(frac=0.3, random_state=SEED)

    # Baseline (dados completos)
    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    base_mia = run_membership_inference_attack(
        df=df_baseline,
        target_model=base_results["results"]["model"],
        X_target_train=base_results["results"]["X_train"],
        y_target_train=base_results["results"]["y_train_true"],
        X_target_test=base_results["results"]["X_test"],
        y_target_test=base_results["results"]["y_test_true"],
        build_preprocessor=build_preprocessor,
    )
    base_attack_acc = compute_attack_metrics(
        y_true=base_mia["y_true"],
        y_pred=base_mia["y_pred"]
    )["attack_acc"]

    # Modelo mais complexo em dataset reduzido
    X = df_small.drop(columns=["salario"])
    y = df_small["salario"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    preprocessor = build_preprocessor(df=df_small)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    overfit_model = clone(base_results["results"]["model"])
    if hasattr(overfit_model, "estimators_"):
        overfit_model.set_params(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2
        )
    elif hasattr(overfit_model, "get_booster"):
        overfit_model.set_params(
            n_estimators=400,
            max_depth=6,
            min_child_weight=1
        )

    overfit_model.fit(X_train, y_train)
    overfit_model.preprocessor_ = preprocessor

    overfit_mia = run_membership_inference_attack(
        df=df_small,
        target_model=overfit_model,
        X_target_train=X_train,
        y_target_train=y_train,
        X_target_test=X_test,
        y_target_test=y_test,
        build_preprocessor=build_preprocessor,
    )
    overfit_attack_acc = compute_attack_metrics(
        y_true=overfit_mia["y_true"],
        y_pred=overfit_mia["y_pred"]
    )["attack_acc"]

    status = "ok" if overfit_attack_acc > base_attack_acc else "suspeito"
    return {
        "test": "overfitting_amplificado",
        "attack_acc": overfit_attack_acc,
        "baseline_acc": base_attack_acc,
        "status": status,
        "debug": overfit_mia.get("debug"),
    }


def test_balanceamento_dataset_ataque(df_baseline, model_runner, model_name):
    """
    TESTE 5 - BALANCEAMENTO DO DATASET DE ATAQUE
    Verifica proporcao de membros vs nao-membros no target e no shadow (estimado).
    """
    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    mia_results = run_membership_inference_attack(
        df=df_baseline,
        target_model=base_results["results"]["model"],
        X_target_train=base_results["results"]["X_train"],
        y_target_train=base_results["results"]["y_train_true"],
        X_target_test=base_results["results"]["X_test"],
        y_target_test=base_results["results"]["y_test_true"],
        build_preprocessor=build_preprocessor,
    )

    y_target = np.asarray(mia_results["y_true"])
    target_ratio = float(np.mean(y_target))

    target_test_size = _get_target_test_size(
        base_results["results"]["X_train"],
        base_results["results"]["X_test"]
    )
    shadow_ratio = 1.0 - target_test_size

    status = "ok" if 0.4 <= target_ratio <= 0.6 and 0.4 <= shadow_ratio <= 0.6 else "suspeito"
    return {
        "test": "balanceamento_dataset_ataque",
        "target_member_ratio": target_ratio,
        "shadow_member_ratio": shadow_ratio,
        "status": status,
    }


def test_consistencia_features(df_baseline, model_runner, model_name):
    """
    TESTE 6 - CONSISTENCIA DE FEATURES
    Compara estatisticas basicas entre features de shadow e target.
    """
    base_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    target_model = base_results["results"]["model"]
    X_target_train = base_results["results"]["X_train"]
    X_target_test = base_results["results"]["X_test"]
    y_target_train = base_results["results"]["y_train_true"]
    y_target_test = base_results["results"]["y_test_true"]

    # Cria um shadow simples para comparacao estatistica
    shadow_df, _ = train_test_split(
        df_baseline, test_size=0.5, random_state=SEED
    )
    X_shadow_raw = shadow_df.drop(columns=["salario"])
    y_shadow = shadow_df["salario"]

    target_test_size = _get_target_test_size(X_target_train, X_target_test)
    X_s_train_raw, X_s_test_raw, y_s_train, y_s_test = train_test_split(
        X_shadow_raw, y_shadow, test_size=target_test_size, random_state=SEED
    )

    preprocessor = getattr(target_model, "preprocessor_", None)
    if preprocessor is None:
        preprocessor = build_preprocessor(df=shadow_df)
        X_s_train = preprocessor.fit_transform(X_s_train_raw)
        X_s_test = preprocessor.transform(X_s_test_raw)
    else:
        X_s_train = preprocessor.transform(X_s_train_raw)
        X_s_test = preprocessor.transform(X_s_test_raw)

    shadow_model = clone(target_model)
    if "random_state" in shadow_model.get_params():
        shadow_model.set_params(random_state=SEED)
    shadow_model.fit(X_s_train, y_s_train)

    X_shadow_features = np.vstack([
        _extract_features(shadow_model, X_s_train, y_s_train),
        _extract_features(shadow_model, X_s_test, y_s_test),
    ])

    X_target_features = np.vstack([
        _extract_features(target_model, X_target_train, y_target_train),
        _extract_features(target_model, X_target_test, y_target_test),
    ])

    if X_shadow_features.shape[1] != X_target_features.shape[1]:
        return {
            "test": "consistencia_features",
            "status": "inconsistente",
            "reason": "numero_colunas_diferente",
        }

    shadow_stats = _compute_feature_stats(X_shadow_features)
    target_stats = _compute_feature_stats(X_target_features)

    mean_diff = abs(shadow_stats["mean"] - target_stats["mean"])
    std_diff = abs(shadow_stats["std"] - target_stats["std"])

    mean_rel = mean_diff / (abs(target_stats["mean"]) + 1e-9)
    std_rel = std_diff / (abs(target_stats["std"]) + 1e-9)

    status = "ok" if mean_rel <= 0.5 and std_rel <= 0.5 else "inconsistente"
    return {
        "test": "consistencia_features",
        "shadow_mean": shadow_stats["mean"],
        "target_mean": target_stats["mean"],
        "shadow_std": shadow_stats["std"],
        "target_std": target_stats["std"],
        "mean_rel_diff": mean_rel,
        "std_rel_diff": std_rel,
        "status": status,
    }


def run_all_mia_sanity_checks(df_baseline, model_runner, model_name):
    results = []
    results.append(test_random_label(df_baseline, model_runner, model_name))
    results.append(test_train_vs_train(df_baseline, model_runner, model_name))
    results.append(test_train_vs_noise(df_baseline, model_runner, model_name))
    results.append(test_overfitting_amplificado(df_baseline, model_runner, model_name))
    results.append(test_balanceamento_dataset_ataque(df_baseline, model_runner, model_name))
    results.append(test_consistencia_features(df_baseline, model_runner, model_name))
    return results


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn.preprocessing._encoders"
    )

    df_baseline, *_ = load_data()

    model_configs = [
        ("XGBoost", run_xgboost),
        ("Random Forest", run_random_forest),
        ("Gradient Boosting", run_gradient_boosting),
        ("Extra Trees", run_extra_trees),
    ]

    for model_name, runner in model_configs:
        print(f"\n{'='*40}")
        print(f"{model_name} - MIA sanity validation")
        print(f"{'='*40}")
        results = run_all_mia_sanity_checks(
            df_baseline=df_baseline,
            model_runner=runner,
            model_name=model_name
        )
        pprint.pprint(results)
