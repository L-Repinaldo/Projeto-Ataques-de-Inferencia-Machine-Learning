from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor
from metrics import compute_utility_metrics

TARGET_COL = "salario"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.3


@dataclass
class ModelOutputs:
    model: Any
    preprocessor: Any
    X_train: Any
    X_test: Any
    X_train_raw: Any
    X_test_raw: Any
    y_train_true: Any
    y_train_pred: Any
    y_test_true: Any
    y_test_pred: Any
    utility: Dict[str, Any]


def split_dataset(df, *, target: str = TARGET_COL, test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )


def run_model_and_collect(
    df,
    model_runner: Callable[..., Dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    target: str = TARGET_COL,
) -> ModelOutputs:
    """
    Runs a model using the project preprocessor and collects the outputs
    needed by sanity checks.
    """
    preprocessor = build_preprocessor(df=df)

    run_output = model_runner(
        df=df,
        preprocessor=preprocessor,
        seed=seed,
        test_size=test_size,
        target=target,
    )

    X_train_raw, X_test_raw, _, _ = split_dataset(
        df, target=target, test_size=test_size, seed=seed
    )

    model = run_output["model"]
    fitted_preprocessor = getattr(model, "preprocessor_", preprocessor)

    X_train = fitted_preprocessor.transform(X_train_raw)
    X_test = fitted_preprocessor.transform(X_test_raw)

    utility = compute_utility_metrics(
        y_train_true=run_output["y_train_true"],
        y_train_pred=run_output["y_train_pred"],
        y_test_true=run_output["y_test_true"],
        y_test_pred=run_output["y_test_pred"],
    )

    return ModelOutputs(
        model=model,
        preprocessor=fitted_preprocessor,
        X_train=X_train,
        X_test=X_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train_true=run_output["y_train_true"],
        y_train_pred=run_output["y_train_pred"],
        y_test_true=run_output["y_test_true"],
        y_test_pred=run_output["y_test_pred"],
        utility=utility,
    )


def attack_inputs_from_utility(utility: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train_abs_error": np.asarray(utility["train_abs_error"]),
        "test_abs_error": np.asarray(utility["test_abs_error"]),
    }
