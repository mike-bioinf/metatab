from typing import TypeAlias, Union

from metatab.estimators.estimators.xgb import (
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

from metatab.estimators.estimators.catboost import (
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier
)

from metatab.estimators.estimators.rf import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyEnsembledRandomForestClassifier,
    MetaTuneRandomForestClassifier,
    MetaEnsembleRandomForestClassifier
)

from metatab.estimators.estimators.extra_trees import (
    MyExtraTreesClassifier,
    MyTunedExtraTreesClassifier,
    MyEnsembledExtraTreesClassifier,
    MetaTuneExtraTreesClassifier,
    MetaEnsembleExtraTreesClassifier
)

from metatab.estimators.estimators.lgbm import (
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

from metatab.estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier,
    MetaTuneTabPFNClassifier,
    MetaEnsembleTabPFNClassifier
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
    MyEnsembledTabPFNClassifier
]


DefaultEstimator: TypeAlias = Union[
    MyRandomForestClassifier,
    MyExtraTreesClassifier,
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
    MyEnsembledExtraTreesClassifier,
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
    MyTunedExtraTreesClassifier,
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
    "MyExtraTreesClassifier",
    "MyTunedExtraTreesClassifier",
    "MyEnsembledExtraTreesClassifier",
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
    "Estimator",
    "DefaultEstimator",
    "TunedEstimator",
    "EnsembledEstimator"
]