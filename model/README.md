# ML Training & Evaluation Pipeline

This directory contains the Python code that trains, evaluates, and exports a
linear regression model whose job is to **predict relative simulation speedup**
for each candidate module that Essent could deduplicate. The exported
coefficients are consumed at compile time by the Scala runtime
(`MLRankModel.scala`).

## Why this pipeline exists

The Essent dedup compiler can only deduplicate **one module per compile**. The
naive heuristic picks whichever module removes the most IR nodes (the "benefit
rank"). That metric is cheap to compute but unreliable: removing more IR nodes
does not always translate to faster simulation. Runtime effects — cross-instance
coupling, cache pressure, boundary-signal overhead — can make a low-benefit
module outperform the highest-benefit one in practice.

This pipeline exists to **replace that guess with a data-driven prediction**.
By training a linear model on features extracted from the compiler's own
statement graph (things a heuristic could never easily combine), we let the
compiler score every candidate module and pick the one most likely to produce
the fastest simulator binary. The payoff is measurable: better module selection
→ faster generated simulators → shorter hardware-design iteration cycles.

Without this pipeline, every time the design landscape changes (new RTL
generators, larger SoCs, different memory hierarchies) the heuristic silently
degrades. A trained model can be retrained on fresh data, adapting
automatically.

Crucially, the pipeline does not just train a model and hope for the best — it
includes a head-to-head comparison against the heuristic on every evaluation
run. If the ML model cannot demonstrably out-rank the heuristic on held-out
designs, it should not be deployed. This comparison is the central output of
`evaluate_ranking.py` and is described in detail in the
[metrics glossary](#how-we-compare-the-ml-model-against-the-heuristic) below.

## Directory layout

| File | Purpose |
|------|---------|
| `config.py` | Defines `FEATURE_COLS` (the 7 graph-level features), `GROUP_COLS` (design / benchmark / parallel_cpus — used to group rows during cross-validation and ranking evaluation), and `TARGET_COL` (`relative_speedup`). |
| `pipeline.py` | Builds a scikit-learn pipeline of `StandardScaler` → `LinearRegression`. Provides `extract_raw_coefficients` to convert standardized weights back to raw-feature space, and `export_coefficients` to write a two-line CSV (header + values) that the Scala `MLRankModel.loadCoefficients()` can parse directly. |
| `metrics.py` | Regression metrics (`R²`, MAE, RMSE, median AE) plus ranking metrics (`ndcg_at_k` — top-k overlap, and `ranking_report` which computes per-group NDCG@1 and Kendall's τ). Also provides `dummy_baseline_mae` (predict-the-mean) for sanity checking. |
| `cross_validation.py` | `regression_cv`: Grouped k-fold CV by design when ≥ 2 designs exist, otherwise plain k-fold. `ranking_cv_leave_one_design_out`: Hold one design out at a time, refit, predict, and compute ranking metrics on the held-out groups — the most honest estimate of generalisation to an unseen design. |
| `train_and_evaluate.py` | **CLI entry point for training.** Usage: `python train_and_evaluate.py <input_csv> <output_csv>`. Fits on the full dataset, prints in-sample regression metrics plus CV results, then writes the raw-space coefficient CSV that should be copied into `src/main/resources/META-INF/ml-rank-coefficients.csv`. |
| `evaluate_ranking.py` | **CLI entry point for ranking evaluation.** Usage: `python evaluate_ranking.py <input_csv>`. Compares the heuristic ranking (benefit order) against the ML model on NDCG@1 and Kendall's τ, both in-sample and via leave-one-design-out CV. |
| `regression_dataset.csv` | Example / working training data. Each row is one (design, rank, benchmark, parallel_cpus) tuple with the 7 features and `relative_speedup` label collected from actual simulation runs. |

## Metrics glossary

The pipeline produces two families of metrics. Understanding what each one
means — and why it matters for module selection — is essential for interpreting
results and deciding whether to deploy updated coefficients.

### Regression metrics (computed by `train_and_evaluate.py`)

These measure how close the model's predicted speedup values are to the true
values. They answer: **"How accurately do we predict the number?"**

**R² (coefficient of determination)** — The fraction of variance in
`relative_speedup` that the model explains. R² = 1 is a perfect fit; R² = 0
means the model is no better than always predicting the mean; negative values
mean it is actively worse. *In this project:* R² is the single most important
check before exporting coefficients. If cross-validated R² is near zero, the
features do not carry enough signal and the model should not be deployed.

**MAE (mean absolute error)** — The average of |predicted − true| across all
samples. Expressed in the same units as `relative_speedup`, so a MAE of 0.04
means predictions are off by ~4 percentage points on average. *In this project:*
compare MAE against the dummy baseline (see below). If the model's MAE is not
lower, the learned coefficients add no value over predicting the mean.

**RMSE (root mean squared error)** — Square root of the mean of
(predicted − true)². Like MAE, but penalises large errors disproportionately
because they are squared before averaging. *In this project:* when RMSE is much
larger than MAE, the model has a few very bad outlier predictions. Worth
investigating which designs or modules cause them.

**Median AE (median absolute error)** — The median (not mean) of |predicted −
true|. Unlike MAE, a single catastrophic prediction does not inflate it.
*In this project:* if MAE is high but Median AE is low, most predictions are
fine and only a handful are poor — which may be acceptable when the ultimate
goal is just to rank the top module correctly.

**Dummy baseline MAE** — MAE of a model that always predicts the mean of
`relative_speedup`, ignoring all features. *In this project:* a sanity check.
If the trained model cannot beat this trivially simple baseline, the features
are not informative and the model is useless.

### Ranking metrics (computed by `evaluate_ranking.py`)

In practice the compiler does not need an exact speedup number — it needs to
pick the **best** module from a set of candidates. Ranking metrics measure
selection quality directly.

**NDCG@k (normalised discounted cumulative gain at k)** — Compares the top-*k*
items in the predicted ranking against the top-*k* items in the true ranking.
For k = 1 (the default in this project), it reduces to a simple question: *did
the model's #1 pick match the truly best module?* The score is 1 if yes, 0 if
no, averaged over all groups. *In this project:* NDCG@1 is the most
decision-relevant metric. An NDCG@1 of 0.80 means the model picks the optimal
module in 80% of (design, benchmark, parallel_cpus) groups. The compiler only
deduplicates its top pick, so getting position #1 right is what matters most.

**Kendall's τ (tau)** — A rank-correlation coefficient measuring agreement
between two orderings. It counts the number of concordant vs. discordant pairs:
τ = +1 means the predicted and true orderings match perfectly, τ = 0 means
they are uncorrelated (random), τ = −1 means they are perfectly inverted.
*In this project:* even when the top pick is wrong, a high τ means the
predicted ordering is mostly correct — so the second-best candidate is likely
still a reasonable choice. A negative τ on the heuristic baseline confirms that
the IR-node-removal heuristic actively mis-ranks modules for runtime speedup.

### How we compare the ML model against the heuristic

The whole point of this pipeline is to answer one question: **does the ML model
pick better modules than the heuristic the compiler already has?** The
comparison is built into `evaluate_ranking.py` and works as follows.

**What the heuristic is.** The existing compiler ranks candidate modules by
"dedup benefit" — `(instance_count - 1) × module_ir_size` — and picks the
top-ranked one. This ordering is captured in the `rank` column of the dataset:
rank 1 is the heuristic's first choice, rank 2 its second choice, and so on.
To evaluate the heuristic with the same ranking machinery used for the ML
model, `evaluate_ranking.py` converts `rank` into a sortable score by negating
it (`heuristic_score = -rank`), so that sorting descending by score reproduces
the heuristic's original ordering.

**What the ML model does differently.** Instead of ranking by IR-node removal,
the ML model predicts `relative_speedup` for each candidate module using the
7 structural features, then ranks by predicted speedup (highest first). The
module with the highest predicted speedup is the ML model's pick.

**The three-level comparison.** `evaluate_ranking.py` prints results in three
blocks, each computing NDCG@1 and Kendall's τ:

1. **Heuristic baseline** — ranking metrics using the `rank` column as the
   predicted ordering. This is the status quo: how often does the current
   compiler pick the truly best module?
2. **ML model (in-sample)** — ranking metrics using predicted speedup from a
   model trained on all the data. This is an optimistic upper bound on ML
   performance (the model has seen the test data during training).
3. **ML model (leave-one-design-out CV)** — ranking metrics using predicted
   speedup from a model that has *never seen the test design*. This is the
   realistic estimate of how the ML model will perform in production.

**Reading the "gap" output.** At the end, the script prints the difference
`ML metric − heuristic metric` for both NDCG@1 and Kendall's τ. A positive
gap means the ML model outperforms the heuristic; a negative gap means it is
worse. For example:

```
NDCG@1 gap (ML - heuristic) = +0.0833
Tau gap    (ML - heuristic) = +0.4926
```

This says the ML model picks the best module in ~8% more groups than the
heuristic, and its full ordering correlates much better with reality (τ gap of
+0.49). The CV gap is the one that matters for deployment — if the CV gap is
positive, the ML model is expected to outperform the heuristic on new, unseen
designs.

**When to deploy.** If the CV NDCG@1 gap is positive (or at least zero) and
the CV Kendall's τ gap is positive, the ML model is an improvement and the
exported coefficients should be deployed. If the CV gaps are negative, the
heuristic is still better and the model needs more training data or different
features before it is useful.

### Cross-validation strategies

In-sample metrics (train and evaluate on the same data) are optimistically
biased. Cross-validation (CV) gives a realistic estimate of how the model will
perform on data it has never seen.

**Grouped k-fold CV** — The dataset is split into *k* folds such that all rows
from a given *design* are kept together in the same fold. Each fold is held out
once as the test set while the model trains on the remaining *k − 1* folds.
*In this project:* used by `train_and_evaluate.py` for regression metrics. By
grouping on design, we ensure the model is tested on wholly unseen designs, not
just unseen rows from a familiar design. Without grouping, the same design
leaking into both train and test sets produces misleadingly high scores.

**Plain k-fold CV** — Rows are randomly assigned to *k* folds regardless of
design. *In this project:* a fallback used only when there is a single design
in the dataset (grouped CV is impossible). Less rigorous because the same
design appears in both train and test sets.

**Leave-one-design-out CV** — A special case of grouped CV where *k* equals the
number of designs. Each design is held out one at a time; the model is retrained
from scratch for each fold and ranking metrics are computed on the held-out
design's groups. *In this project:* used by `evaluate_ranking.py`. This is the
strictest test of generalisation: "If a brand-new design comes along that I have
never trained on, will the model still rank its modules correctly?"

The gap between in-sample and CV metrics reveals overfitting. If in-sample R²
is 0.99 but CV R² is 0.2, the model has memorised the training data and will
not generalise. For a low-capacity model like `StandardScaler` +
`LinearRegression` with only 7 features, this gap is usually small — but
checking it is essential before deploying new coefficients.

## Features (shared with Scala `ModuleFeatures`)

All features are computed **before** the dedup transform is applied, using the
statement graph and module-instance tables already available in the compiler.
Each one captures a different aspect of *why* deduplicating a particular module
might (or might not) produce faster simulation.

| Feature | Definition | Why it matters |
|---------|------------|----------------|
| `instance_count` | Number of instances of the candidate module in the design. | More instances means more code sharing from dedup, but also more dispatcher overhead at runtime. The relationship with speedup is non-linear. |
| `module_ir_size` | Number of IR statements inside one instance of the module. | Larger modules offer more code to share but are also harder to keep in instruction cache after dedup merges them. |
| `boundary_signal_count` | Nodes with at least one edge crossing the instance boundary. | A proxy for coupling cost. High boundary counts mean dedup requires more bookkeeping to shuttle data in and out of the shared function, which can negate the code-size savings. |
| `boundary_to_interior_ratio` | `boundary_signal_count / original_ir_size`. | Normalises boundary cost relative to design size. A module with 100 boundary signals in a 500-node design is much more coupled than the same module in a 50,000-node design. |
| `edge_count_within` | Total outgoing edges that stay inside the instance subgraph. | Captures internal computational complexity. Modules with dense internal connectivity tend to benefit more from dedup because the shared evaluation function amortises that complexity. |
| `fraction_design_covered` | `(instance_count × module_ir_size) / original_ir_size`. | What share of the entire design this dedup would affect. Deduplicating a module that covers 60% of the design has a very different impact than one covering 2%. |
| `original_ir_size` | Total valid IR nodes in the design. | Contextualises all other features. The same absolute boundary count means different things in a small design vs. a large one. |

## Typical workflow

```bash
# 1. Install dependencies (Python ≥ 3.8)
cd model/
pip install numpy pandas scikit-learn scipy   # or use `uv sync` from the repo root

# 2. Train and export coefficients
python train_and_evaluate.py regression_dataset.csv ../src/main/resources/META-INF/ml-rank-coefficients.csv

# 3. Evaluate ranking quality
python evaluate_ranking.py regression_dataset.csv

# 4. Rebuild the Essent JAR so the Scala runtime picks up the new coefficients
cd .. && sbt assembly
```

## Extending the dataset

Each Essent compile with dedup enabled writes a `dedup_features.csv` next to its
output. Combine those with simulation throughput measurements and a
`relative_speedup` column (`median_throughput / baseline_throughput`) to grow
`regression_dataset.csv`. Rerun the training pipeline above to refresh
coefficients.
