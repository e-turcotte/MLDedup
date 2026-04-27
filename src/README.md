# Scala Compiler Integration for ML-Based Dedup Ranking

This directory contains the Essent compiler source. The files below were
**added or modified** to support ML-guided module deduplication selection.

## Goal

Essent's dedup pass merges duplicate module instances to reduce simulation work,
but it can only deduplicate **the top ranked modules per compile**. The baseline strategy
always picked the module with the greatest static IR-node reduction. The
additions here let the compiler optionally use a trained linear model to predict
which module yields the best **runtime speedup**, and also emit telemetry so
the Python training pipeline can be fed new labelled data.

## Changed / added files

### `ArgsParser.scala`

- Added `mlRank: Boolean` to `OptFlags`.
- Added the `--ml-rank` CLI flag. When set, it also forces `dedup = true` and
  `useCondParts = true`. Help text: *"use ML model to select dedup module
  (ignores baked-in rank)"*.

### `MLRankModel.scala` (new)

Houses all ML inference logic, fully self-contained.

| Component | Description |
|-----------|-------------|
| `ModuleFeatures` | Case class holding the 5 engineered features (`boundaryToInteriorRatio`, `logOriginalIRSize`, `instanceCount`, `boundaryRatioXInstanceCount`, `hasBoundary`). The field order in `toArray` matches the Python `FEATURE_COLS` and the coefficients CSV column order. |
| `loadCoefficients()` | Reads `META-INF/ml-rank-coefficients.csv` from the classpath. Expects a header row and one data row with `intercept` followed by the 5 feature weights. Returns `Option[Array[Double]]`. |
| `predict(coeffs, features)` | Simple dot product: `intercept + Σ(coeff_i × feature_i)`. No scaling needed at runtime because the Python export step (`pipeline.extract_raw_coefficients`) converts standardized weights to raw-feature space. |
| `computeFeatures(modName, modInstInfo, sg, originalIRSize)` | Builds a `ModuleFeatures` from the statement graph **before** dedup. Boundary count = number of nodes with at least one edge crossing the instance subgraph boundary (a proxy for the coupling overhead dedup will introduce). |
| `selectBestModule(candidates, …)` | Scores every candidate module (those with `instanceCount > 1`), takes `maxBy` predicted score, and returns `(bestModuleName, pseudoRank)` where `pseudoRank` is the 1-based position in the old benefit-sorted list — useful for logging and comparing against the heuristic. |

### `Compiler.scala`

Three changes in `EssentEmitter.execute`:

1. **`object EssentEmitter.readDedupRankFromResource()`** — reads an integer
   from `META-INF/essent-dedup-rank`. Rank 0 means "skip dedup"; ranks 1–10
   pick that position in the benefit-sorted module list (clamped to available
   modules). This mechanism lets you build JAR variants (essent-1.jar …
   essent-10.jar) that each deduplicate a different module, producing training
   rows for `regression_dataset.csv`.

2. **Three-way module selection** replaces the old "always pick `head`":
   - `--ml-rank` → load coefficients, filter to modules with > 1 instance,
     call `selectBestModule`, use `pseudoRank` for logging.
   - Resource rank = 0 or no modules → skip dedup entirely.
   - Otherwise → use baked-in rank (deterministic sweeps / training-data
     collection).

3. **`dedup_features.csv` emission** — after the dedup planning phase, writes
   one CSV row next to the compiler output with timestamp, design name, rank
   used, chosen module, and all 7 raw feature columns. This row can be merged with
   simulation throughput to extend the training dataset.

### Resource files

| File | Content |
|------|---------|
| `main/resources/META-INF/essent-dedup-rank` | Currently `0` (no dedup in non-ML mode). Change to 1–10 for sweep builds. |
| `main/resources/META-INF/ml-rank-coefficients.csv` | Placeholder zeros. Replace with output from `model/train_and_evaluate.py` for real predictions. `--ml-rank` will throw if the file is missing or malformed. |

## End-to-end compile with ML ranking

```bash
# Build the JAR
sbt assembly

# Compile a FIRRTL design with ML-guided dedup
./essent --ml-rank -O3 my_design.fir

# The compiler will:
#   1. Enumerate all multiply-instantiated modules.
#   2. Compute 5 engineered features for each candidate.
#   3. Score candidates with the linear model from ml-rank-coefficients.csv.
#   4. Deduplicate the module with the highest predicted speedup.
#   5. Write dedup_features.csv alongside the output .h file.
```
