from typing import Literal


# Type of estimators that can be tuned.
ESTIMATOR_TYPE = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]