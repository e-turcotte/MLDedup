import sys
import pandas as pd
from sklearn.linear_model import LinearRegression

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

model = LinearRegression()
model.fit(X, y)

r2 = model.score(X, y)
print(f"R² = {r2:.6f}")
print(f"Intercept: {model.intercept_:.10f}")
for name, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {name}: {coef:.10f}")

header = "intercept," + ",".join(FEATURE_COLS)
values = ",".join([f"{model.intercept_}"] + [str(c) for c in model.coef_])

with open(output_csv, "w") as f:
    f.write(header + "\n")
    f.write(values + "\n")

print(f"\nWrote coefficients to {output_csv}")
