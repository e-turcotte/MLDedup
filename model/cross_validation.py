import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_validate

from metrics import hit_at_k
from pipeline import build_pipeline

from scipy.stats import kendalltau


def regression_cv(X, y, groups=None):
    """Run grouped or plain k-fold CV with regression scoring."""
    n_samples = len(y)
    n_groups = groups.nunique() if groups is not None else 0

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
        "neg_median_ae": "neg_median_absolute_error",
    }

    if n_samples >= 3 and n_groups >= 2:
        n_splits = n_groups
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=groups))
        description = (
            f"Leave-one-design-out grouped CV ({n_splits} folds; "
            "each fold holds out exactly one design)"
        )
    elif n_samples >= 5:
        n_splits = min(5, n_samples)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = cv.split(X, y)
        description = (
            f"KFold CV ({n_splits} folds; rows may share design across train/test)\n"
            "    (Add a 'design' column to use grouped CV by design.)"
        )
    else:
        return None

    cv_res = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=splits,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "r2": cv_res["test_r2"],
        "mae": -cv_res["test_neg_mae"],
        "rmse": np.sqrt(-cv_res["test_neg_mse"]),
        "median_ae": -cv_res["test_neg_median_ae"],
        "description": description,
    }


def ranking_cv_leave_one_design_out(df, feature_cols, group_cols):
    """Leave-one-design-out CV evaluated with ranking metrics."""
    designs = df["design"].unique()
    if len(designs) < 2:
        return None

    hit1_scores = []
    tau_scores = []

    for held_out in designs:
        train = df[df["design"] != held_out]
        test = df[df["design"] == held_out].copy()

        cv_pipe = build_pipeline()
        cv_pipe.fit(train[feature_cols], train["relative_speedup"])
        test["cv_predicted"] = cv_pipe.predict(test[feature_cols])

        for _, group in test.groupby(group_cols):
            if len(group) < 2:
                continue
            true_order = group.sort_values("relative_speedup", ascending=False)["rank"].tolist()
            pred_order = group.sort_values("cv_predicted", ascending=False)["rank"].tolist()
            hit1_scores.append(hit_at_k(true_order, pred_order, k=1))
            tau, _ = kendalltau(true_order, pred_order)
            if not np.isnan(tau):
                tau_scores.append(tau)

    return {
        "hit1_scores": hit1_scores,
        "tau_scores": tau_scores,
        "mean_hit1": np.mean(hit1_scores),
        "mean_tau": np.mean(tau_scores),
        "std_tau": np.std(tau_scores),
    }
