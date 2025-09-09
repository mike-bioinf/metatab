from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from finetabpfn import AesFineTunedTabPFNClassifier



Classifier: TypeAlias = Union[
    RandomForestClassifier,
    XGBClassifier,
    CatBoostClassifier,
    TabPFNClassifier, 
    AutoTabPFNClassifier, 
    AesFineTunedTabPFNClassifier
]


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