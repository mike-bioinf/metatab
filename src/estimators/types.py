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