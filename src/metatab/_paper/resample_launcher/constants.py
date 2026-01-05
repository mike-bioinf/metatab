CPU_ESTIMATORS = [
    "random_forest",
    "xgb",
    "es_xgb",
    "catboost",
    "es_catboost",
    "lgbm",
    "es_lgbm",
    "extra_trees"
]

# we use just a single gpu to allow comparisons even when multiple can be used
SINGLE_GPU_ESTIMATORS = [
    "tabpfn",
    "realmlp",
    "tabm"
]

MULTIPLE_GPU_ESTIMATORS = []

ALL_GPU_ESTIMATORS = SINGLE_GPU_ESTIMATORS + MULTIPLE_GPU_ESTIMATORS