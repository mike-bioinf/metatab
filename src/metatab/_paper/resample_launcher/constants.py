CPU_ESTIMATORS = [
    "random_forest",
    "xgb",
    "es_xgb",
    "catboost",
    "es_catboost",
    "lgbm",
    "es_lgbm"
]

SINGLE_GPU_ESTIMATORS = [
    "tabpfn"
]

MULTIPLE_GPU_ESTIMATORS = []

ALL_GPU_ESTIMATORS = SINGLE_GPU_ESTIMATORS + MULTIPLE_GPU_ESTIMATORS