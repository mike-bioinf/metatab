from typing import TypeAlias, Union

from estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier 
)

from estimators.catboost import (
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier
)

from estimators.rf import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier
)

from estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier
)

from estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyAutoTabPFNClassifier
)



Estimator: TypeAlias = Union[
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyAutoTabPFNClassifier
]

__all__ = [
    "MyRandomForestClassifier",
    "MyTunedRandomForestClassifier",
    "MyXGBClassifier",
    "MyESXGBClassifier",
    "MyTunedXGBClassifier", 
    "MyTunedESXGBClassifier",
    "MyCatBoostClassifier",
    "MyESCatBoostClassifier",
    "MyTunedCatBoostClassifier",
    "MyTunedESCatBoostClassifier",
    "MyLGBMClassifier",
    "MyESLGBMClassifier",
    "MyTunedLGBMClassifier",
    "MyTunedESLGBMClassifier"
    "MyTabPFNClassifier",
    "MyTunedTabPFNClassifier",
    "MyAutoTabPFNClassifier",
    "Estimator"
]