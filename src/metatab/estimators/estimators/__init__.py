from typing import TypeAlias, Union

from metatab.estimators.estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier
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
    MyEnsembledRandomForestClassifier
)

from metatab.estimators.estimators.extra_trees import (
    MyExtraTreesClassifier,
    MyTunedExtraTreesClassifier,
    MyEnsembledExtraTreesClassifier
)

from metatab.estimators.estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier
)

from metatab.estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier
)

from metatab.estimators.estimators.realmlp import (
    MyRealMLPClassifier,
    MyTunedRealMLPClassifier,
    MyEnsembledRealMLPClassifier
)

from metatab.estimators.estimators.tabm import (
    MyTabMClassifier,
    MyTunedTabMClassifier,
    MyEnsembledTabMClassifier
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
    MyEnsembledRealMLPClassifier,
    MyTabMClassifier,
    MyTunedTabMClassifier,
    MyEnsembledTabMClassifier,
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
    MyRealMLPClassifier,
    MyTabMClassifier
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
    MyEnsembledRealMLPClassifier,
    MyEnsembledTabMClassifier
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
    MyTunedRealMLPClassifier,
    MyTunedTabMClassifier
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
    "MyTabMClassifier",
    "MyTunedTabMClassifier",
    "MyEnsembledTabMClassifier",
    "Estimator",
    "DefaultEstimator",
    "TunedEstimator",
    "EnsembledEstimator"
]