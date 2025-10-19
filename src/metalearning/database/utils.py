from __future__ import annotations

import joblib
from typing import TYPE_CHECKING
from importlib import resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from estimators.constants import Classifier



SURROGATE_DATABASE_PATH = resources.files("metalearning.database.database")

SURROGATE_DATABASE = {
    "random_forest": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_rf.joblib"),
    "xgb": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_xgboost.joblib"),
    "catboost": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_catboost.joblib"),
    "lgbm": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_lgbm.joblib"),
    "tabpfn": SURROGATE_DATABASE_PATH.joinpath("surrogate_framework_for_tabpfn.joblib"),
}


def query_surrogate_framework(clf_or_pipe: Classifier | Pipeline) -> Pipeline:
    '''
    Retrieve the fitted surrogate framework corresponding to a classifier.

    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Either a classifier instance or a pipeline whose last step is a Classifier.

    Returns:
        Pipeline: 
        The surrogate framework, i.e. the pipeline of preprocessing steps 
        plus the surrogate model.
    '''
    clf = clf_or_pipe[-1] if isinstance(clf_or_pipe, Pipeline) else clf_or_pipe

    if isinstance(clf, RandomForestClassifier):
        return load_surrogate_framework(SURROGATE_DATABASE["random_forest"])
    elif isinstance(clf, XGBClassifier):
        return load_surrogate_framework(SURROGATE_DATABASE["xgb"])
    elif isinstance(clf, CatBoostClassifier):
        return load_surrogate_framework(SURROGATE_DATABASE["catboost"])
    elif isinstance(clf, LGBMClassifier):
        return load_surrogate_framework(SURROGATE_DATABASE["lgbm"])
    elif isinstance(clf, TabPFNClassifier):
        return load_surrogate_framework(SURROGATE_DATABASE["tabpfn"])
    else:
        raise ValueError(
            "Is not possible to retrieve a surrogate framework based on the classifier type."
        )
    

def load_surrogate_framework(path: Traversable) -> Pipeline:
    return joblib.load(path.open("rb"))