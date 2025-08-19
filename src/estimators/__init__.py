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

from estimators.tabpfn import (
    MyTabPFNClassifier
)



Estimator: TypeAlias = Union[
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyTabPFNClassifier
]


__all__ = [
    "MyXGBClassifier",
    "MyESXGBClassifier",
    "MyTunedXGBClassifier", 
    "MyTunedESXGBClassifier",
    "MyRandomForestClassifier",
    "MyCatBoostClassifier",
    "MyESCatBoostClassifier",
    "MyTunedCatBoostClassifier",
    "MyTunedESCatBoostClassifier",
    "MyTunedRandomForestClassifier",
    "MyTabPFNClassifier",
    "Estimator"
]