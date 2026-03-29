from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pprint
import warnings

from data import load_data
from model import run_xgboost, run_random_forest, run_gradient_boosting, run_extra_trees

from sanity_check.model_checks import run_all_model_sanity_checks


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn.preprocessing._encoders",
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
            model_name=model_name,
        )
        all_results.extend(results)

    print(f"\n{'='*40}")
    print("Model sanity validation")
    print(f"{'='*40}")
    pprint.pprint(all_results)
