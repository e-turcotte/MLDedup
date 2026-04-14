import sys

import pandas as pd

from config import FEATURE_COLS, TARGET_COL
from cross_validation import regression_cv
from metrics import dummy_baseline_mae, regression_report
from pipeline import export_coefficients, extract_raw_coefficients, fit_pipeline

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_csv> <output_csv>")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

df = pd.read_csv(input_csv)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

n_samples, n_features = X.shape
print(f"n_samples={n_samples}, n_features={n_features}")

# --- Full fit (in-sample) ---------------------------------------------------

pipe = fit_pipeline(X, y)
y_pred = pipe.predict(X)

report = regression_report(y, y_pred)
baseline_mae = dummy_baseline_mae(X, y)

print("\n--- Full fit (in-sample; can be optimistic) ---")
print(f"R²              = {report['r2']:.6f}")
print(f"MAE             = {report['mae']:.6f}")
print(f"RMSE            = {report['rmse']:.6f}")
print(f"Median |error| = {report['median_ae']:.6f}")
print(f"Dummy (mean y) MAE = {baseline_mae:.6f}  (baseline; lower is better)")

# --- Standardized coefficients ----------------------------------------------

model = pipe.named_steps["linearregression"]
print("\n--- Standardized coefficients (per-σ effect on relative_speedup) ---")
for name, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {name}: {coef:.10f}")

# --- Cross-validation -------------------------------------------------------

groups = df["design"] if "design" in df.columns else None
cv_result = regression_cv(X, y, groups)

if cv_result is not None:
    print(f"\n--- {cv_result['description']} ---")
    print(f"R²   : {cv_result['r2'].mean():.6f} ± {cv_result['r2'].std():.6f}")
    print(f"MAE  : {cv_result['mae'].mean():.6f} ± {cv_result['mae'].std():.6f}")
    print(f"RMSE : {cv_result['rmse'].mean():.6f} ± {cv_result['rmse'].std():.6f}")
else:
    print("\n--- CV skipped (need ≥3 rows and ≥2 designs for grouped CV, or ≥5 rows for KFold) ---")

# --- Export raw-space coefficients ------------------------------------------

raw_intercept, coef_dict = export_coefficients(pipe, FEATURE_COLS, output_csv)

print("\n--- Raw-space coefficients (written to output CSV) ---")
print(f"Intercept: {raw_intercept:.10f}")
for name, coef in coef_dict.items():
    print(f"  {name}: {coef:.10f}")

print(f"\nWrote coefficients to {output_csv}")
