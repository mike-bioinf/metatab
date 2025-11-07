from __future__ import annotations

import joblib
from typing import TYPE_CHECKING
from importlib import resources
from pathlib import Path
from sklearn.pipeline import Pipeline
from estimators.types import TUNABLE_ESTIMATOR_TYPE
from estimators.constants import TUNABLE_ESTIMATORS

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable



def set_surrogate_database(database: None | dict[str, str]) -> None:
    '''
    Set globally the surrogate database as "_SURROGATE_DATABASE" object.
    '''
    global _SURROGATE_DATABASE
    
    if database is None:
        _SURROGATE_DATABASE = return_internal_database()
    else:
        _check_keys_database(database)
        # Path and Traversable objects are functionally equivalent (duck typing)
        _SURROGATE_DATABASE = {k: Path(v) for k, v in database.items()}
        _check_model_file_existence(_SURROGATE_DATABASE)



def _check_keys_database(database: dict) -> None:
    for estimator in database.keys():
        if estimator not in TUNABLE_ESTIMATORS:
            raise KeyError(f"'{estimator}' key is unvalid.")



def _check_model_file_existence(database: dict[str, Path]) -> None:
    for k, v in database.items():
        if not v.exists():
            raise ValueError(f"The file pointed by the key '{k}' does not exists.")



def return_internal_database() -> dict[str, Traversable]:
    database_path = resources.files("metalearning.database.database")
    return {
        "random_forest": database_path.joinpath("surrogate_framework_for_rf.joblib"),
        "xgb": database_path.joinpath("surrogate_framework_for_xgb.joblib"),
        "es_xgb": database_path.joinpath("surrogate_framework_for_es_xgb.joblib"),
        "catboost": database_path.joinpath("surrogate_framework_for_catboost.joblib"),
        "es_catboost": database_path.joinpath("surrogate_framework_for_es_catboost.joblib"),
        "lgbm": database_path.joinpath("surrogate_framework_for_lgbm.joblib"),
        "es_lgbm": database_path.joinpath("surrogate_framework_for_es_lgbm.joblib"),
        "tabpfn": database_path.joinpath("surrogate_framework_for_tabpfn.joblib")
    }



### By default we set/use the internal database
_SURROGATE_DATABASE = return_internal_database()



def query_surrogate_framework(type_estimator: TUNABLE_ESTIMATOR_TYPE) -> Pipeline:
    '''Retrieve the fitted surrogate pipeline given the type_estimator'''
    return load_surrogate_framework(_SURROGATE_DATABASE[type_estimator]) 



def load_surrogate_framework(path: Traversable) -> Pipeline:
    return joblib.load(path.open("rb"))