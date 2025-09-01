from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from finetabpfn import AesFineTunedTabPFNClassifier



Classifier: TypeAlias = Union[
    RandomForestClassifier,
    XGBClassifier,
    CatBoostClassifier,
    TabPFNClassifier, 
    AutoTabPFNClassifier, 
    AesFineTunedTabPFNClassifier
]

TabPFNClassifiers: TypeAlias = Union[
    TabPFNClassifier
]

# list of estimators employing early stop on a validation set 
# and requiring the early_stop_rounds informagion 
EARLY_STOPPED_ESTIMATORS = [
    "es_xgb",
    "es_catboost",
    "es_lgbm"
]

# list of non tunable estimators
NON_TUNABLE_ESTIMATORS = [
    "autotabpfn"
]