from estimators.types import Estimator

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
