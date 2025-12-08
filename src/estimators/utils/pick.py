from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from estimators.estimators.xgb import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier
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
    MyEnsembledRandomForestClassifier
)

from estimators.estimators.lgbm import (
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier
)

from estimators.estimators.tabpfn import (
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
)

if TYPE_CHECKING:
    from estimators.estimators import Estimator
    from estimators.utils.types import EstimatorType



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
        
        case ("xgb", "default"):
            return MyXGBClassifier
        case ("xgb", "tune"):
            return MyTunedXGBClassifier
        case ("es_xgb", "default"):
            return MyESXGBClassifier
        case ("es_xgb", "tune"):
            return MyTunedESXGBClassifier
        
        case("catboost", "default"):
            return MyCatBoostClassifier
        case("catboost", "tune"):
            return MyTunedCatBoostClassifier
        case ("es_catboost", "default"):
            return MyESCatBoostClassifier
        case("es_catboost", "tune"):
            return MyTunedESCatBoostClassifier
        
        case ("lgbm", "default"):
            return MyLGBMClassifier
        case("lgbm", "tune"):
            return MyTunedLGBMClassifier
        case ("es_lgbm", "default"):
            return MyESLGBMClassifier
        case ("es_lgbm", "tune"):
            return MyTunedESLGBMClassifier

        case ("tabpfn", "default"):
            return MyTabPFNClassifier
        case("tabpfn", "tune"):
            return MyTunedTabPFNClassifier
        # case("autotabpfn", _):
        #     return MyAutoTabPFNClassifier
        # case("finetunetabpfn", _):
        #     return MyAesFineTunedTabPFNClassifier
    
        case _:
            raise ValueError("Unrecognized estimator-mode combination.")