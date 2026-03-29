from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pprint
import warnings

from data import load_data
from model import run_xgboost, run_random_forest, run_gradient_boosting, run_extra_trees

from sanity_check.common import run_model_and_collect, attack_inputs_from_utility
from sanity_check.mia_checks import run_all_mia_sanity_checks


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

    for model_name, runner in model_configs:
        print(f"\n{'='*40}")
        print(f"{model_name} - MIA sanity validation")
        print(f"{'='*40}")

        outputs = run_model_and_collect(df_baseline, runner)
        target_outputs = attack_inputs_from_utility(outputs.utility)
        results = run_all_mia_sanity_checks(target_outputs)
        pprint.pprint(results)
