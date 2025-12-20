from typing import Literal
from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier


Classifier: TypeAlias = Union[
    RandomForestClassifier,
    ExtraTreesClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier,
    TabPFNClassifier
]


GBDTClassifier: TypeAlias = Union[
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier
]


TunableEstimatorType = Literal[
    "random_forest",
    "extra_trees",
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
    "extra_trees",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


EstimatorType = Literal[
    "random_forest",
    "extra_trees",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]