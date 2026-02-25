from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import pprint
import warnings

from data import load_data
from experiments import run_model, model_metrics
from model import (
    run_random_forest,
    run_xgboost
)
from metrics import compute_utility_metrics


def sanity_check(baseline_results: dict) -> dict:
    train_mae = compute_utility_metrics(
        y_true=baseline_results['results']["y_train_true"],
        y_pred=baseline_results['results']["y_train_pred"],
    )['mae']

    test_mae = compute_utility_metrics(
        y_true=baseline_results['results']["y_test_true"],
        y_pred=baseline_results['results']["y_test_pred"],
    )['mae']

    return {
        'train_mae': train_mae,
        'test_mae': test_mae
    }


def run_machine_learning_sanity_check(model_runner, model_name, df_baseline):
    model_results = run_model(
        name="baseline",
        df=df_baseline,
        model_name=model_name,
        model_runner=model_runner
    )

    sanity = sanity_check(baseline_results=model_results)

    return sanity


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn.preprocessing._encoders"
    )

    df_baseline, *_ = load_data()

    experiments = [
        ("XGBoost", run_xgboost),
        ("Random Forest", run_random_forest),
    ]

    utility_rows = []

    for model_name, runner in experiments:
        print(f"\n{'='*40}")
        print(f"{model_name} sanity check")
        print(f"{'='*40}")

        sanity = run_machine_learning_sanity_check(
            model_runner=runner,
            model_name=model_name,
            df_baseline=df_baseline
        )

        utility_rows.append({
            "model": model_name,
            "dataset": "baseline",
            "train_mae": sanity["train_mae"],
            "test_mae": sanity["test_mae"],
        })

    df_utility_sanity_check = pd.DataFrame(utility_rows)

    print(f"\n{'='*40}")
    print("Sanity check result")
    print(f"{'='*40}")
    pprint.pprint(df_utility_sanity_check)