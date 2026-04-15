FEATURE_COLS = [
    "instance_count",
    "module_ir_size",
    "boundary_signal_count",
    "boundary_to_interior_ratio",
    "edge_count_within",
    "fraction_design_covered",
    "original_ir_size",
]

# Groups for training dataset - determines cross-validation rows
GROUP_COLS = ["design", "benchmark", "parallel_cpus"]

# Target column for training dataset - determines the target variable
TARGET_COL = "relative_speedup"
