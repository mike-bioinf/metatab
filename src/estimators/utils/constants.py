EARLY_STOPPED_ESTIMATORS = [
    "es_xgb",
    "es_catboost",
    "es_lgbm"
]


NON_EARLY_STOPPED_ESTIMATORS = [
    "random_forest",
    "xgb",
    "lgbm",
    "catboost",
    "tabpfn"
]


NON_EARLY_STOPPED_CPU_ESTIMATORS = [
    "random_forest",
    "xgb",
    "lgbm",
    "catboost"
]


NON_EARLY_STOPPED_GPU_ESTIMATORS = [
    "tabpfn"
]


# NON_TUNABLE_ESTIMATORS = []


# PCA_INCOMPATIBLE_ESTIMATORS = []


GBDT_ESTIMATORS = [
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


TUNABLE_ESTIMATORS = [
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]


ALL_ESTIMATORS = [
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]