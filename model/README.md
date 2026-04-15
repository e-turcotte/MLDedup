# ML Training & Evaluation Pipeline

This directory contains the Python code that trains, evaluates, and exports a
linear regression model whose job is to **predict relative simulation speedup**
for each candidate module that Essent could deduplicate. The exported
coefficients are consumed at compile time by the Scala runtime
(`MLRankModel.scala`).

## Goal

The Essent dedup compiler can only deduplicate **one module per compile**. The
naive heuristic picks whichever module removes the most IR nodes (the "benefit
rank"). That ignores runtime effects such as cross-instance coupling and cache
behaviour. This pipeline learns from labelled simulation runs so the compiler
can make a better choice.

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

## Features (shared with Scala `ModuleFeatures`)

All features are computed **before** the dedup transform is applied, using the
statement graph and module-instance tables already available in the compiler.

| Feature | Description |
|---------|-------------|
| `instance_count` | Number of instances of the candidate module in the design. |
| `module_ir_size` | Number of IR statements inside one instance of the module. |
| `boundary_signal_count` | Nodes with at least one edge crossing the instance boundary — proxy for coupling cost. |
| `boundary_to_interior_ratio` | `boundary_signal_count / original_ir_size`. |
| `edge_count_within` | Total outgoing edges that stay inside the instance subgraph — captures internal complexity. |
| `fraction_design_covered` | `(instance_count × module_ir_size) / original_ir_size` — what share of the design this dedup would affect. |
| `original_ir_size` | Total valid IR nodes in the design (contextualises the other features). |

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
