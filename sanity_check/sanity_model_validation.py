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
from metrics import compute_utility_metrics
from model import run_xgboost, run_random_forest, run_gradient_boosting, run_extra_trees


SEED = 42
TEST_SIZE = 0.3


def _mae(y_true, y_pred):
    return compute_utility_metrics(y_true=y_true, y_pred=y_pred)["mae"]


def _rmse(y_true, y_pred):
    return compute_utility_metrics(y_true=y_true, y_pred=y_pred)["rmse"]


def _status_by_expected(condition_ok):
    return "ok" if condition_ok else "suspeito"


def test_random_label(df_baseline, model_runner, model_name):
    """
    TESTE 1 - RANDOM LABEL TEST
    Verifica se o modelo colapsa quando nao ha relacao entre X e y.
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
    X_test = base_results["results"]["X_test"]
    y_train = base_results["results"]["y_train_true"]
    y_test = base_results["results"]["y_test_true"]

    y_train_shuffled = rng.permutation(y_train)

    shuffled_model = clone(base_results["results"]["model"])
    shuffled_model.fit(X_train, y_train_shuffled)

    y_test_pred = shuffled_model.predict(X_test)
    mae = _mae(y_test, y_test_pred)

    # Espera-se piora em relacao ao baseline
    base_mae = _mae(
        base_results["results"]["y_test_true"],
        base_results["results"]["y_test_pred"]
    )

    status = _status_by_expected(mae > base_mae * 1.1)
    return {
        "model": model_name,
        "test": "random_label",
        "mae": mae,
        "baseline_mae": base_mae,
        "status": status,
    }


def test_overfitting_controlado(df_baseline, model_runner, model_name):
    """
    TESTE 2 - OVERFITTING CONTROLADO
    Forca o modelo a overfitar aumentando complexidade e reduzindo dados.
    """
    df_small = df_baseline.sample(frac=0.3, random_state=SEED)

    model_results = run_model(
        name="baseline",
        df=df_small,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    model = clone(model_results["results"]["model"])
    if hasattr(model, "estimators_"):
        model.set_params(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2
        )
    elif hasattr(model, "get_booster"):
        model.set_params(
            n_estimators=400,
            max_depth=6,
            min_child_weight=1
        )

    X_train = model_results["results"]["X_train"]
    X_test = model_results["results"]["X_test"]
    y_train = model_results["results"]["y_train_true"]
    y_test = model_results["results"]["y_test_true"]

    model.fit(X_train, y_train)

    train_mae = _mae(y_train, model.predict(X_train))
    test_mae = _mae(y_test, model.predict(X_test))
    gap = test_mae - train_mae

    status = _status_by_expected(gap > 0.1 * train_mae)
    return {
        "model": model_name,
        "test": "overfitting_controlado",
        "train_mae": train_mae,
        "test_mae": test_mae,
        "mae_gap": gap,
        "status": status,
    }


def test_underfitting_controlado(df_baseline, model_runner, model_name):
    """
    TESTE 3 - UNDERFITTING CONTROLADO
    Reduz complexidade para verificar perda de capacidade.
    """
    model_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    model = clone(model_results["results"]["model"])
    if hasattr(model, "estimators_"):
        model.set_params(
            n_estimators=50,
            max_depth=2,
            min_samples_leaf=30,
            min_samples_split=60
        )
    elif hasattr(model, "get_booster"):
        model.set_params(
            n_estimators=50,
            max_depth=2,
            min_child_weight=50
        )

    X_train = model_results["results"]["X_train"]
    X_test = model_results["results"]["X_test"]
    y_train = model_results["results"]["y_train_true"]
    y_test = model_results["results"]["y_test_true"]

    model.fit(X_train, y_train)
    train_mae = _mae(y_train, model.predict(X_train))
    test_mae = _mae(y_test, model.predict(X_test))
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


def test_data_leakage(df_baseline, model_runner, model_name):
    """
    TESTE 4 - DATA LEAKAGE CHECK
    Verifica se o erro de treino e menor que o erro de teste.
    """
    model_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    train_mae = _mae(
        model_results["results"]["y_train_true"],
        model_results["results"]["y_train_pred"]
    )
    test_mae = _mae(
        model_results["results"]["y_test_true"],
        model_results["results"]["y_test_pred"]
    )

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
    TESTE 5 - STABILITY ACROSS SEEDS
    Mede variancia de MAE em diferentes seeds.
    """
    seeds = [42, 123, 2026, 77, 99]
    maes = []

    for seed in seeds:
        model_results = run_model(
            name="baseline",
            df=df_baseline,
            model_name=model_name,
            model_runner=lambda **kwargs: model_runner(
                **kwargs,
                seed=seed,
                test_size=TEST_SIZE
            ),
        )
        maes.append(_mae(
            model_results["results"]["y_test_true"],
            model_results["results"]["y_test_pred"]
        ))

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


def test_sensibilidade_tamanho_dataset(df_baseline, model_runner, model_name):
    """
    TESTE 6 - SENSIBILIDADE AO TAMANHO DO DATASET
    Verifica se MAE melhora com mais dados.
    """
    # Mantem o mesmo conjunto de teste para evitar vies de amostragem
    X = df_baseline.drop(columns=["salario"])
    y = df_baseline["salario"]
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Usa o preprocessor do modelo baseline para consistencia
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
    preprocessor = base_results["results"]["model"].preprocessor_
    X_test = preprocessor.transform(X_test_full)

    fracs = [0.3, 0.6, 1.0]
    results = []

    for frac in fracs:
        if frac >= 1.0:
            X_train_part = X_train_full
            y_train_part = y_train_full
        else:
            X_train_part, _, y_train_part, _ = train_test_split(
                X_train_full, y_train_full, train_size=frac, random_state=SEED
            )
        X_train = preprocessor.transform(X_train_part)

        model = clone(base_results["results"]["model"])
        if "random_state" in model.get_params():
            model.set_params(random_state=SEED)
        model.fit(X_train, y_train_part)

        y_test_pred = model.predict(X_test)
        mae = _mae(y_test_full, y_test_pred)
        results.append((frac, mae))

    # Espera-se melhoria (MAE menor) com mais dados, tolerando pequenas variacoes
    status = _status_by_expected(results[0][1] >= results[1][1] * 0.98 and results[1][1] >= results[2][1] * 0.98)
    return {
        "model": model_name,
        "test": "sensibilidade_tamanho_dataset",
        "mae_30": results[0][1],
        "mae_60": results[1][1],
        "mae_100": results[2][1],
        "status": status,
    }


def test_feature_importance(df_baseline, model_runner, model_name):
    """
    TESTE 7 - FEATURE IMPORTANCE CHECK
    Verifica se a distribuicao de importancia nao e degenerada.
    """
    model_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=SEED,
            test_size=TEST_SIZE
        ),
    )

    model = model_results["results"]["model"]
    preprocessor = getattr(model, "preprocessor_", None)
    feature_names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_booster"):
        score = model.get_booster().get_score(importance_type="gain")
        if not score:
            return {
                "model": model_name,
                "test": "feature_importance",
                "status": "suspeito",
                "reason": "sem_importancia",
            }
        importances = np.array(list(score.values()), dtype=float)
    else:
        return {
            "model": model_name,
            "test": "feature_importance",
            "status": "suspeito",
            "reason": "indisponivel",
        }

    total = float(np.sum(importances)) + 1e-9
    shares = importances / total
    max_share = float(np.max(importances)) / total
    top3_share = float(np.sort(importances)[-3:].sum()) / total

    top_features = []
    if feature_names and len(feature_names) == len(importances):
        top_idx = np.argsort(shares)[-5:][::-1]
        top_features = [
            {"feature": feature_names[i], "share": float(shares[i])}
            for i in top_idx
        ]
    else:
        top_idx = np.argsort(shares)[-5:][::-1]
        top_features = [
            {"feature": f"f{i}", "share": float(shares[i])}
            for i in top_idx
        ]

    # Critera: nao pode ter concentracao extrema em uma unica feature,
    # mas permite dominancia moderada quando o dataset e assimetrico.
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
    results = []
    results.append(test_random_label(df_baseline, model_runner, model_name))
    results.append(test_overfitting_controlado(df_baseline, model_runner, model_name))
    results.append(test_underfitting_controlado(df_baseline, model_runner, model_name))
    results.append(test_data_leakage(df_baseline, model_runner, model_name))
    results.append(test_stability_across_seeds(df_baseline, model_runner, model_name))
    results.append(test_sensibilidade_tamanho_dataset(df_baseline, model_runner, model_name))
    results.append(test_feature_importance(df_baseline, model_runner, model_name))
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

    all_results = []
    for model_name, runner in model_configs:
        results = run_all_model_sanity_checks(
            df_baseline=df_baseline,
            model_runner=runner,
            model_name=model_name
        )
        all_results.extend(results)

    print(f"\n{'='*40}")
    print("Model sanity validation")
    print(f"{'='*40}")
    pprint.pprint(all_results)
