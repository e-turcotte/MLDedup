import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, cross_validate
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

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_csv> <output_csv>")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

df = pd.read_csv(input_csv)

X = df[FEATURE_COLS]
y = df["relative_speedup"]

n_samples, n_features = X.shape
print(f"n_samples={n_samples}, n_features={n_features}")

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X, y)
y_pred = pipe.predict(X)

scaler = pipe.named_steps["standardscaler"]
model = pipe.named_steps["linearregression"]

r2_in = r2_score(y, y_pred)
mae_in = mean_absolute_error(y, y_pred)
rmse_in = np.sqrt(mean_squared_error(y, y_pred))
medae_in = median_absolute_error(y, y_pred)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X, y)
dummy_mae = mean_absolute_error(y, dummy.predict(X))

print("\n--- Full fit (in-sample; can be optimistic) ---")
print(f"R²              = {r2_in:.6f}")
print(f"MAE             = {mae_in:.6f}")
print(f"RMSE            = {rmse_in:.6f}")
print(f"Median |error| = {medae_in:.6f}")
print(f"Dummy (mean y) MAE = {dummy_mae:.6f}  (baseline; lower is better)")

print("\n--- Standardized coefficients (per-σ effect on relative_speedup) ---")
for name, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {name}: {coef:.10f}")

# Cross-validation: prefer leave-designs-out when possible
scoring = {
    "r2": "r2",
    "neg_mae": "neg_mean_absolute_error",
    "neg_mse": "neg_mean_squared_error",
}
groups = df["design"] if "design" in df.columns else None
n_groups = groups.nunique() if groups is not None else 0

if n_samples >= 3 and n_groups >= 2:
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=groups))
    print(f"\n--- Grouped CV by design ({n_splits} folds; tests generalization to unseen designs) ---")
elif n_samples >= 5:
    n_splits = min(5, n_samples)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    splits = cv.split(X, y)
    print(f"\n--- KFold CV ({n_splits} folds; rows may share design across train/test) ---")
    print("    (Add a 'design' column to use grouped CV by design.)")
else:
    splits = None

if splits is not None:
    cv_res = cross_validate(
        make_pipeline(StandardScaler(), LinearRegression()),
        X,
        y,
        cv=splits,
        scoring=scoring,
        n_jobs=-1,
    )
    r2_cv = cv_res["test_r2"]
    mae_cv = -cv_res["test_neg_mae"]
    rmse_cv = np.sqrt(-cv_res["test_neg_mse"])
    print(f"R²   : {r2_cv.mean():.6f} ± {r2_cv.std():.6f}")
    print(f"MAE  : {mae_cv.mean():.6f} ± {mae_cv.std():.6f}")
    print(f"RMSE : {rmse_cv.mean():.6f} ± {rmse_cv.std():.6f}")
else:
    print("\n--- CV skipped (need ≥3 rows and ≥2 designs for grouped CV, or ≥5 rows for KFold) ---")

# Convert standardized coefficients back to raw-feature space for the Scala runtime:
#   raw_coef[i] = std_coef[i] / σ[i]
#   raw_intercept = std_intercept - Σ(std_coef[i] * μ[i] / σ[i])
raw_coefs = model.coef_ / scaler.scale_
raw_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

print("\n--- Raw-space coefficients (written to output CSV) ---")
print(f"Intercept: {raw_intercept:.10f}")
for name, coef in zip(FEATURE_COLS, raw_coefs):
    print(f"  {name}: {coef:.10f}")

header = "intercept," + ",".join(FEATURE_COLS)
values = ",".join([f"{raw_intercept}"] + [str(c) for c in raw_coefs])

with open(output_csv, "w") as f:
    f.write(header + "\n")
    f.write(values + "\n")

print(f"\nWrote coefficients to {output_csv}")
