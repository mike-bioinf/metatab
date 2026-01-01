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

from metatab.estimators.estimators.realmlp import (
    MyRealMLPClassifier,
    MyTunedRealMLPClassifier,
    MyEnsembledRealMLPClassifier,
    MetaTuneRealMLPClassifier,
    MetaEnsembleRealMLPClassifier
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
    MyRealMLPClassifier,
    MyTunedRealMLPClassifier,
    MyEnsembledRealMLPClassifier
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
    MyTabPFNClassifier,
    MyRealMLPClassifier
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
    MyEnsembledTabPFNClassifier,
    MyEnsembledRealMLPClassifier
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
    MyTunedTabPFNClassifier,
    MyTunedRealMLPClassifier
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
    "MyRealMLPClassifier",
    "MyTunedRealMLPClassifier",
    "MyEnsembledRealMLPClassifier",
    "Estimator",
    "DefaultEstimator",
    "TunedEstimator",
    "EnsembledEstimator"
]