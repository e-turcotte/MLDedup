import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_validate

from metrics import ndcg_at_k
from pipeline import build_pipeline

from scipy.stats import kendalltau


def regression_cv(X, y, groups=None):
    """Run grouped or plain k-fold CV with regression scoring.

    Returns ``None`` if there are too few samples/groups, otherwise a dict
    with arrays ``r2``, ``mae``, ``rmse`` (one value per fold).
    Also returns a ``description`` string for display purposes.
    """
    n_samples = len(y)
    n_groups = groups.nunique() if groups is not None else 0

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    if n_samples >= 3 and n_groups >= 2:
        n_splits = min(5, n_groups)
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=groups))
        description = f"Grouped CV by design ({n_splits} folds; tests generalization to unseen designs)"
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
        "description": description,
    }


def ranking_cv_leave_one_design_out(df, feature_cols, group_cols):
    """Leave-one-design-out CV evaluated with ranking metrics.

    Returns ``None`` if fewer than 2 designs are present, otherwise a dict
    with arrays ``ndcg1_scores`` and ``tau_scores`` plus their means.
    """
    designs = df["design"].unique()
    if len(designs) < 2:
        return None

    ndcg1_scores = []
    tau_scores = []

    for held_out in designs:
        train = df[df["design"] != held_out]
        test = df[df["design"] == held_out].copy()

        cv_pipe = build_pipeline()
        cv_pipe.fit(train[feature_cols], train["relative_speedup"])
        test["cv_predicted"] = cv_pipe.predict(test[feature_cols])

        for _, group in test.groupby(group_cols):
            true_order = group.sort_values("relative_speedup", ascending=False)["rank"].tolist()
            pred_order = group.sort_values("cv_predicted", ascending=False)["rank"].tolist()
            ndcg1_scores.append(ndcg_at_k(true_order, pred_order, k=1))
            tau, _ = kendalltau(true_order, pred_order)
            tau_scores.append(tau)

    return {
        "ndcg1_scores": ndcg1_scores,
        "tau_scores": tau_scores,
        "mean_ndcg1": np.mean(ndcg1_scores),
        "mean_tau": np.mean(tau_scores),
        "std_tau": np.std(tau_scores),
    }
