import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "instance_count",
    "module_ir_size",
    "boundary_signal_count",
    "boundary_to_interior_ratio",
    "edge_count_within",
    "fraction_design_covered",
    "original_ir_size",
]

GROUP_COLS = ["design", "benchmark", "parallel_cpus"]


def ndcg_at_k(true_order, pred_order, k=1):
    return len(set(true_order[:k]) & set(pred_order[:k])) / k


def ranking_metrics(df, speedup_col, label):
    """Compute per-group NDCG@1 and Kendall's tau using `speedup_col` as the predicted ranking."""
    ndcg1_scores = []
    tau_scores = []

    for _, group in df.groupby(GROUP_COLS):
        true_order = group.sort_values("relative_speedup", ascending=False)["rank"].tolist()
        pred_order = group.sort_values(speedup_col, ascending=False)["rank"].tolist()

        ndcg1_scores.append(ndcg_at_k(true_order, pred_order, k=1))

        tau, _ = kendalltau(true_order, pred_order)
        tau_scores.append(tau)

    mean_ndcg1 = np.mean(ndcg1_scores)
    mean_tau = np.mean(tau_scores)
    print(f"\n--- {label} ---")
    print(f"NDCG@1         = {mean_ndcg1:.4f}  ({sum(ndcg1_scores):.0f}/{len(ndcg1_scores)} groups picked the best module)")
    print(f"Kendall's tau  = {mean_tau:.4f} ± {np.std(tau_scores):.4f}")
    return mean_ndcg1, mean_tau


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_csv>")
    sys.exit(1)

input_csv = sys.argv[1]
df = pd.read_csv(input_csv)

for col in GROUP_COLS + ["rank", "relative_speedup"]:
    if col not in df.columns:
        print(f"ERROR: required column '{col}' not found in {input_csv}")
        sys.exit(1)

X = df[FEATURE_COLS]
y = df["relative_speedup"]

n_groups = df.groupby(GROUP_COLS).ngroups
print(f"n_samples={len(df)}, n_groups={n_groups}")

# Heuristic baseline: rank column is already 1=best-benefit, 2=second, etc.
# Lower rank number = heuristic's preferred module.
# To sort descending by "heuristic score", invert rank.
df["heuristic_score"] = -df["rank"]
heuristic_ndcg1, heuristic_tau = ranking_metrics(df, "heuristic_score", "Heuristic (dedup-benefit ranking)")

# ML model (in-sample)
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X, y)
df["predicted_speedup"] = pipe.predict(X)
model_ndcg1, model_tau = ranking_metrics(df, "predicted_speedup", "ML model (in-sample)")

# Leave-one-design-out CV with ranking metrics
designs = df["design"].unique()
if len(designs) >= 2:
    cv_ndcg1_scores = []
    cv_tau_scores = []

    for held_out in designs:
        train = df[df["design"] != held_out]
        test = df[df["design"] == held_out].copy()

        cv_pipe = make_pipeline(StandardScaler(), LinearRegression())
        cv_pipe.fit(train[FEATURE_COLS], train["relative_speedup"])
        test["cv_predicted"] = cv_pipe.predict(test[FEATURE_COLS])

        for _, group in test.groupby(GROUP_COLS):
            true_order = group.sort_values("relative_speedup", ascending=False)["rank"].tolist()
            pred_order = group.sort_values("cv_predicted", ascending=False)["rank"].tolist()
            cv_ndcg1_scores.append(ndcg_at_k(true_order, pred_order, k=1))
            tau, _ = kendalltau(true_order, pred_order)
            cv_tau_scores.append(tau)

    cv_ndcg1 = np.mean(cv_ndcg1_scores)
    cv_tau = np.mean(cv_tau_scores)
    print(f"\n--- ML model (leave-one-design-out CV) ---")
    print(f"NDCG@1         = {cv_ndcg1:.4f}  ({sum(cv_ndcg1_scores):.0f}/{len(cv_ndcg1_scores)} groups picked the best module)")
    print(f"Kendall's tau  = {cv_tau:.4f} ± {np.std(cv_tau_scores):.4f}")
else:
    cv_ndcg1 = None
    cv_tau = None
    print("\n--- CV skipped (need ≥2 designs) ---")

# Headline comparison
print("\n--- Comparison (in-sample) ---")
print(f"NDCG@1 gap (ML - heuristic) = {model_ndcg1 - heuristic_ndcg1:+.4f}")
print(f"Tau gap    (ML - heuristic) = {model_tau - heuristic_tau:+.4f}")
if cv_ndcg1 is not None:
    print(f"\n--- Comparison (leave-one-design-out CV) ---")
    print(f"NDCG@1 gap (ML - heuristic) = {cv_ndcg1 - heuristic_ndcg1:+.4f}")
    print(f"Tau gap    (ML - heuristic) = {cv_tau - heuristic_tau:+.4f}")
