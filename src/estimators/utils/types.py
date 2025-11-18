from typing import Literal
from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

# circular import error
# from estimators.tabpfn import SingleDatasetAesFineTunedTabpfnClassifier



Classifier: TypeAlias = Union[
    RandomForestClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier,
    TabPFNClassifier, 
    AutoTabPFNClassifier, 
    #SingleDatasetAesFineTunedTabpfnClassifier
]


GBDTClassifier: TypeAlias = Union[
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier
]


TunableEstimatorType = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]


EsEstimatorType: Literal[
    "es_lgbm",
    "es_xgb",
    "es_catboost"
]


GBDTEstimatorType = Literal[
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


TreeEstimatorType = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


EstimatorType = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn",
    "autotabpfn",
    "finetunetabpfn"
]


PreprocessingStrategy = Literal[
    "estimator_default",
    "base",
    "density_filter",
    "pca",
    "no"
]