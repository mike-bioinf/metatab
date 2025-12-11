from typing import TypeAlias, Union

from estimators.estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier,
    MetaEnsembleXGBClassifier,
    MetaEnsembleEsXGBClassifier
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
    MetaTuneRandomForestClassifier,
    MetaEnsembleRandomForestClassifier
)

from estimators.estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier,
    MetaEnsembleLGBMClassifier,
    MetaEnsembleEsLGBMClassifier
)

from estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier,
    MetaTuneTabPFNClassifier,
    MetaEnsembleTabPFNClassifier
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


DefaultEstimator: TypeAlias = Union[
    MyRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTabPFNClassifier
]


EnsembledEstimator: TypeAlias = Union[
    MyEnsembledRandomForestClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MyEnsembledTabPFNClassifier
]


TunedEstimator: TypeAlias = Union[
    MyTunedRandomForestClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyTunedTabPFNClassifier
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
    "DefaultEstimator",
    "TunedEstimator",
    "EnsembledEstimator"
]