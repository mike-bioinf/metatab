from __future__ import annotations

import joblib
from typing import TYPE_CHECKING
from importlib import resources
from sklearn.pipeline import Pipeline
from hp_search.constants import ESTIMATOR_TYPE

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable



SURROGATE_DATABASE_PATH = resources.files("metalearning.database.database")


SURROGATE_DATABASE = {
    "random_forest": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_rf.joblib"),
    "xgb": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_xgboost.joblib"),
    "es_xgb": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_es_xgboost.joblib"),
    "catboost": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_catboost.joblib"),
    "es_catboost": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_es_catboost.joblib"),
    "lgbm": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_lgbm.joblib"),
    "es_lgbm": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_es_lgbm.joblib"),
    "tabpfn": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_tabpfn.joblib"),
}


def query_surrogate_framework(type_estimator: ESTIMATOR_TYPE) -> Pipeline:
    '''Retrieve the fitted surrogate pipeline give the input type_estimator'''
    return load_surrogate_framework(SURROGATE_DATABASE[type_estimator]) 

    
def load_surrogate_framework(path: Traversable) -> Pipeline:
    return joblib.load(path.open("rb"))