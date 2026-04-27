import numpy as np
from scipy.stats import kendalltau
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_report(y_true, y_pred):
    """Return a dict of standard regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "median_ae": median_absolute_error(y_true, y_pred),
    }


def dummy_baseline_mae(X, y):
    """MAE of a predict-the-mean baseline."""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X, y)
    return mean_absolute_error(y, dummy.predict(X))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def hit_at_k(true_order, pred_order, k=1):
    """Top-k set overlap between true and predicted orderings (fraction of top-k items shared)."""
    return len(set(true_order[:k]) & set(pred_order[:k])) / k


def ranking_report(df, speedup_col, group_cols):
    """Compute per-group hit@1 and Kendall's tau."""
    hit1_scores = []
    tau_scores = []

    for _, group in df.groupby(group_cols):
        if len(group) < 2:
            continue
        true_order = group.sort_values("relative_speedup", ascending=False)["rank"].tolist()
        pred_order = group.sort_values(speedup_col, ascending=False)["rank"].tolist()

        hit1_scores.append(hit_at_k(true_order, pred_order, k=1))

        tau, _ = kendalltau(true_order, pred_order)
        if np.isnan(tau):
            continue
        tau_scores.append(tau)

    return {
        "hit1_scores": hit1_scores,
        "tau_scores": tau_scores,
        "mean_hit1": np.mean(hit1_scores),
        "mean_tau": np.mean(tau_scores),
        "std_tau": np.std(tau_scores),
    }
