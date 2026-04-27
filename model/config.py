FEATURE_COLS = [
    "boundary_to_interior_ratio",
    "log_original_ir_size",
    "instance_count",
    "boundary_ratio_x_instance_count",
    "has_boundary",
    "instance_count_x_log_module_ir_size",
]

# Groups for training dataset - determines cross-validation rows
GROUP_COLS = ["design", "benchmark", "parallel_cpus"]

# Target column for training dataset - determines the target variable
TARGET_COL = "relative_speedup"
