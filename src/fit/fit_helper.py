from estimators.types import Estimator

from estimators.estimators.params import (
    RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS,
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_1
)

from estimators import (
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomizedXGBClassifier,
    MyRandomizedESXGBClassifier,
    MyTabPFNClassifier
)



def pick_estimator_class(pars: dict) -> Estimator:
    match (pars["estimator"], pars["tune"]):
        case ("random_forest", False):
            return MyRandomForestClassifier
        case ("random_forest", True):
            return MyRandomizedRandomForestClassifier
        case ("xgb", False):
            return MyXGBClassifier
        case ("xgb", True):
            return MyRandomizedXGBClassifier
        case ("ex_xgb", False):
            return MyESXGBClassifier
        case ("ex_xgb", True):
            return MyRandomizedESXGBClassifier
        case ("tabpfn", _):
            return MyTabPFNClassifier
        case _:
            raise ValueError("Unsupported estimator.")



def pick_hps_configuration(pars: dict) -> dict | None:
    match (pars["estimator"], pars["tune"], pars["hps_configuration"]):
        case ("random_forest", False, None):
            return None
        case ("random_forest", True, None | "c0"):
            return RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS
        case ("xgb" | "es_xgb", False, None):
            return None
        case ("xgb" | "es_xgb", True, None | "c0"):
            return RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS
        case ("xgb" | "es_xgb", True, "c1"):
            return RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_1 
        case _:
            raise ValueError("Unsupported hps tuning scenario.")