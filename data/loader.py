import pandas as pd
from pathlib import Path


def load_data(path):
    return pd.read_csv(path)


def run():
    base = Path("data/datasets/v-2026-01-13_12-34-40")

    df_baseline = load_data(base / "baseline.csv")
    df_dp_01 = load_data(base / "dp_eps_0.1.csv")
    df_dp_05 = load_data(base / "dp_eps_0.5.csv")
    df_dp_10 = load_data(base / "dp_eps_1.0.csv")
    df_dp_20 = load_data(base / "dp_eps_2.0.csv")

    return df_baseline, df_dp_01, df_dp_05, df_dp_10, df_dp_20
