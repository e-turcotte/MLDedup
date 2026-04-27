import sys

import numpy as np
import pandas as pd

from config import FEATURE_COLS, GROUP_COLS, TARGET_COL
from cross_validation import ranking_cv_leave_one_design_out
from data import load_and_prepare
from metrics import ranking_report
from pipeline import fit_pipeline

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_csv>")
    sys.exit(1)

input_csv = sys.argv[1]
df = load_and_prepare(input_csv)

for col in GROUP_COLS + ["rank", TARGET_COL]:
    if col not in df.columns:
        print(f"ERROR: required column '{col}' not found in {input_csv}")
        sys.exit(1)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

n_groups = df.groupby(GROUP_COLS).ngroups
print(f"n_samples={len(df)}, n_groups={n_groups}")

# --- Heuristic baseline -----------------------------------------------------
# rank=1 is the heuristic's top pick; invert so higher = better for sorting.
df["heuristic_score"] = -df["rank"]
heuristic = ranking_report(df, "heuristic_score", GROUP_COLS)

print("\n--- Heuristic (dedup-benefit ranking) ---")
print(f"Hit@1          = {heuristic['mean_hit1']:.4f}  "
      f"({sum(heuristic['hit1_scores']):.0f}/{len(heuristic['hit1_scores'])} groups picked the best module)")
print(f"Kendall's tau  = {heuristic['mean_tau']:.4f} ± {heuristic['std_tau']:.4f}")

# --- ML model (in-sample) ---------------------------------------------------
pipe = fit_pipeline(X, y)
df["predicted_speedup"] = pipe.predict(X)
model = ranking_report(df, "predicted_speedup", GROUP_COLS)

print("\n--- ML model (in-sample) ---")
print(f"Hit@1          = {model['mean_hit1']:.4f}  "
      f"({sum(model['hit1_scores']):.0f}/{len(model['hit1_scores'])} groups picked the best module)")
print(f"Kendall's tau  = {model['mean_tau']:.4f} ± {model['std_tau']:.4f}")

# --- Per-group diagnostic ---------------------------------------------------
print("\n--- Per-group diagnostic (in-sample) ---")
print(f"{'Group':<55} {'Actual Best':<25} {'ML Pick':<25} {'Heuristic Pick':<25}")
print("-" * 130)
for group_key, group in df.groupby(GROUP_COLS):
    label = " / ".join(str(v) for v in group_key)
    actual_best = group.sort_values("relative_speedup", ascending=False).iloc[0]
    ml_pick = group.sort_values("predicted_speedup", ascending=False).iloc[0]
    heur_pick = group.sort_values("heuristic_score", ascending=False).iloc[0]

    marker_ml = "" if ml_pick["dedup_module"] == actual_best["dedup_module"] else " ✗"
    marker_h = "" if heur_pick["dedup_module"] == actual_best["dedup_module"] else " ✗"
    print(f"{label:<55} {actual_best['dedup_module']:<25} {ml_pick['dedup_module']}{marker_ml:<25} {heur_pick['dedup_module']}{marker_h:<25}")

# --- Leave-one-design-out CV ------------------------------------------------
cv = ranking_cv_leave_one_design_out(df, FEATURE_COLS, GROUP_COLS)

if cv is not None:
    print(f"\n--- ML model (leave-one-design-out CV) ---")
    print(f"Hit@1          = {cv['mean_hit1']:.4f}  "
          f"({sum(cv['hit1_scores']):.0f}/{len(cv['hit1_scores'])} groups picked the best module)")
    print(f"Kendall's tau  = {cv['mean_tau']:.4f} ± {cv['std_tau']:.4f}")
else:
    print("\n--- CV skipped (need ≥2 designs) ---")

# --- Headline comparison ----------------------------------------------------
print("\n--- Comparison (in-sample) ---")
print(f"Hit@1 gap (ML - heuristic) = {model['mean_hit1'] - heuristic['mean_hit1']:+.4f}")
print(f"Tau gap   (ML - heuristic) = {model['mean_tau'] - heuristic['mean_tau']:+.4f}")

if cv is not None:
    print(f"\n--- Comparison (leave-one-design-out CV) ---")
    print(f"Hit@1 gap (ML - heuristic) = {cv['mean_hit1'] - heuristic['mean_hit1']:+.4f}")
    print(f"Tau gap   (ML - heuristic) = {cv['mean_tau'] - heuristic['mean_tau']:+.4f}")
