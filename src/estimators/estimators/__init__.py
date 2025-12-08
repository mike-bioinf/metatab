from typing import TypeAlias, Union

from estimators.estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier
)

from estimators.estimators.catboost import (
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier
)

from estimators.estimators.rf import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyEnsembledRandomForestClassifier,
    MetaTuneRandomForestClassifier
)

from estimators.estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier
)

from estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier,
    MetaTuneTabPFNClassifier,
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
)



Estimator: TypeAlias = Union[
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyEnsembledRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier,
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
]


EnsembledEstimator: TypeAlias = Union[
    MyEnsembledRandomForestClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MyEnsembledTabPFNClassifier,
]


__all__ = [
    "MyRandomForestClassifier",
    "MyTunedRandomForestClassifier",
    "MyEnsembledRandomForestClassifier",
    "MyXGBClassifier",
    "MyESXGBClassifier",
    "MyTunedXGBClassifier", 
    "MyTunedESXGBClassifier",
    "MyEnsembledXGBClassifier",
    "MyEnsembledESXGBClassifier",
    "MyCatBoostClassifier",
    "MyESCatBoostClassifier",
    "MyTunedCatBoostClassifier",
    "MyTunedESCatBoostClassifier",
    "MyEnsembledCatBoostClassifier",
    "MyEnsembledESCatBoostClassifier",
    "MyLGBMClassifier",
    "MyESLGBMClassifier",
    "MyTunedLGBMClassifier",
    "MyTunedESLGBMClassifier",
    "MyEnsembledLGBMClassifier",
    "MyEnsembledESLGBMClassifier",
    "MyTabPFNClassifier",
    "MyTunedTabPFNClassifier",
    "MyEnsembledTabPFNClassifier",
    # "MyAutoTabPFNClassifier",
    # "MyAesFineTunedTabPFNClassifier",
    "Estimator",
    "EnsembledEstimator"
]