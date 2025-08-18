from estimators.types import Estimator

from estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyTabPFNClassifier
)



def pick_estimator_class(pars: dict) -> Estimator:
    match (pars["estimator"], pars["tune"]):
        case ("random_forest", False):
            return MyRandomForestClassifier
        case ("random_forest", True):
            return MyTunedRandomForestClassifier
        case ("xgb", False):
            return MyXGBClassifier
        case ("xgb", True):
            return MyTunedXGBClassifier
        case ("es_xgb", False):
            return MyESXGBClassifier
        case ("es_xgb", True):
            return MyTunedESXGBClassifier
        case ("tabpfn", _):
            return MyTabPFNClassifier
        case _:
            raise ValueError("Unsupported estimator.")
