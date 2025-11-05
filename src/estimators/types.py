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



TUNABLE_ESTIMATOR_TYPE = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn"
]


TREE_ESTIMATOR_TYPE = Literal[
    "random_forest",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


ALL_ESTIMATOR_TYPE = Literal[
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