from typing import TypeAlias, Union

from estimators.estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier
)

from estimators.estimators.catboost import (
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier
)

from estimators.estimators.rf import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MetaTuneRandomForestClassifier
)

from estimators.estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier
)

from estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MetaTuneTabPFNClassifier,
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
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
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
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
    # "MyAutoTabPFNClassifier",
    # "MyAesFineTunedTabPFNClassifier",
    "Estimator"
]