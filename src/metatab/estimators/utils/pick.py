from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


if TYPE_CHECKING:
    from metatab.estimators.estimators import Estimator
    from metatab.estimators.utils.types import EstimatorType



def pick_estimator_class(
    estimator: EstimatorType, 
    mode: Literal["default", "tune", "ensemble"]
) -> Estimator:
    match (estimator, mode):
        case ("random_forest", "default"):
            return MyRandomForestClassifier
        case ("random_forest", "tune"):
            return MyTunedRandomForestClassifier
        case ("random_forest", "ensemble"):
            return MyEnsembledRandomForestClassifier
        
        case ("extra_trees", "default"):
            return MyExtraTreesClassifier
        case ("extra_trees", "tune"):
            return MyTunedExtraTreesClassifier
        case ("extra_trees", "ensemble"):
            return MyEnsembledExtraTreesClassifier
        
        case ("xgb", "default"):
            return MyXGBClassifier
        case ("xgb", "tune"):
            return MyTunedXGBClassifier
        case ("xgb", "ensemble"):
            return MyEnsembledXGBClassifier

        case ("es_xgb", "default"):
            return MyESXGBClassifier
        case ("es_xgb", "tune"):
            return MyTunedESXGBClassifier
        case ("es_xgb", "ensemble"):
            return MyEnsembledESXGBClassifier

        case ("catboost", "default"):
            return MyCatBoostClassifier
        case ("catboost", "tune"):
            return MyTunedCatBoostClassifier
        case ("catboost", "ensemble"):
            return MyEnsembledCatBoostClassifier
        
        case ("es_catboost", "default"):
            return MyESCatBoostClassifier
        case ("es_catboost", "tune"):
            return MyTunedESCatBoostClassifier
        case ("es_catboost", "ensemble"):
            return MyEnsembledESCatBoostClassifier

        case ("lgbm", "default"):
            return MyLGBMClassifier
        case ("lgbm", "tune"):
            return MyTunedLGBMClassifier
        case ("lgbm", "ensemble"):
            return MyEnsembledLGBMClassifier
        
        case ("es_lgbm", "default"):
            return MyESLGBMClassifier
        case ("es_lgbm", "tune"):
            return MyTunedESLGBMClassifier
        case ("es_lgbm", "ensemble"):
            return MyEnsembledESLGBMClassifier

        case ("tabpfn", "default"):
            return MyTabPFNClassifier
        case ("tabpfn", "tune"):
            return MyTunedTabPFNClassifier
        case ("tabpfn", "ensemble"):
            return MyEnsembledTabPFNClassifier

        case ("realmlp", "default"):
            return MyRealMLPClassifier
        case ("realmlp", "tune"):
            return MyTunedRealMLPClassifier
        case ("realmlp", "ensemble"):
            return MyEnsembledRealMLPClassifier
        
        case ("tabm", "default"):
            return MyTabMClassifier
        case ("tabm", "tune"):
            return MyTunedTabMClassifier
        case ("tabm", "ensemble"):
            return MyEnsembledTabMClassifier

        case _:
            raise ValueError("Unrecognized estimator-mode combination.")