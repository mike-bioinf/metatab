EARLY_STOPPED_ESTIMATORS = [
    "es_xgb",
    "es_catboost",
    "es_lgbm"
]


NON_TUNABLE_ESTIMATORS = [
    "autotabpfn",
    "finetunetabpfn"
]


PCA_INCOMPATIBLE_ESTIMATORS = [
    "autotabpfn",
    "finetunetabpfn"
]


GBDT_ESTIMATORS = [
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]