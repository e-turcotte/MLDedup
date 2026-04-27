import numpy as np
import pandas as pd

from config import FEATURE_COLS


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Read the regression dataset and compute any engineered features.

    Both train_and_evaluate.py and evaluate_ranking.py call this function
    so feature engineering is guaranteed to be identical.
    """
    df = pd.read_csv(csv_path)

    # --- Engineered features -------------------------------------------------
    # Add new derived columns here; they will be available to both scripts as
    # long as the corresponding names are also listed in config.FEATURE_COLS.
    df["boundary_ratio_x_instance_count"] = (
        df["boundary_to_interior_ratio"] * df["instance_count"]
    )
    df["log_original_ir_size"] = np.log(df["original_ir_size"])
    df["has_boundary"] = (df["boundary_signal_count"] > 0).astype(int)
    df["instance_count_x_log_module_ir_size"] = (
        df["instance_count"] * np.log(df["module_ir_size"] + 1)
    )

    # Verify that all required feature columns are present after engineering.
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns after load_and_prepare: {missing}. "
            "Check config.FEATURE_COLS and the engineered-feature block in data.py."
        )

    return df
