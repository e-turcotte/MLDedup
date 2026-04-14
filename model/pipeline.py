import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return make_pipeline(StandardScaler(), LinearRegression())


def fit_pipeline(X, y):
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def extract_raw_coefficients(pipe, feature_cols):
    """Convert standardized coefficients to raw-feature space for deployment.

    raw_coef[i] = std_coef[i] / sigma[i]
    raw_intercept = std_intercept - sum(std_coef[i] * mu[i] / sigma[i])
    """
    scaler = pipe.named_steps["standardscaler"]
    model = pipe.named_steps["linearregression"]
    raw_coefs = model.coef_ / scaler.scale_
    raw_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    return raw_intercept, dict(zip(feature_cols, raw_coefs))


def export_coefficients(pipe, feature_cols, output_path):
    """Write raw-space coefficients to a CSV consumable by the Scala runtime."""
    raw_intercept, coef_dict = extract_raw_coefficients(pipe, feature_cols)

    header = "intercept," + ",".join(feature_cols)
    values = ",".join([f"{raw_intercept}"] + [str(coef_dict[c]) for c in feature_cols])

    with open(output_path, "w") as f:
        f.write(header + "\n")
        f.write(values + "\n")

    return raw_intercept, coef_dict
