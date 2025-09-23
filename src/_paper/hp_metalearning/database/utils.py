import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from estimators.constants import Classifier




DATABASE = {
    "random_forest": Path("surrogate_pipeline_for_rf.joblib").resolve(),
    "xgb": Path("surrogate_pipeline_for_xgboost.joblib").resolve(),
    "catboost": Path("surrogate_pipeline_for_catboost.joblib").resolve(),
    "lgbm": Path("surrogate_pipeline_for_lgbm.joblib").resolve(),
    "tabpfn": Path("surrogate_pipeline_for_tabpfn.joblib").resolve()
}


def query_surrogate_framework(clf_or_pipe: Classifier | Pipeline) -> Pipeline:
    '''
    Retrieve the fitted surrogate framework corresponding to a classifier.

    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Either a classifier instance or a pipeline whose last step is a Classifier.

    Returns:
        Pipeline: 
        The surrogate framework, i.e. the surrogate model plus 
        the preprocessing pipeline for the given classifier.
    '''
    clf = clf_or_pipe[-1] if isinstance(clf_or_pipe, Pipeline) else clf_or_pipe

    if isinstance(clf, RandomForestClassifier):
        return load_surrogate_framework(DATABASE["random_forest"])
    elif isinstance(clf, XGBClassifier):
        return load_surrogate_framework(DATABASE["xgb"])
    elif isinstance(clf, CatBoostClassifier):
        return load_surrogate_framework(DATABASE["catboost"])
    elif isinstance(clf, LGBMClassifier):
        return load_surrogate_framework(DATABASE["lgbm"])
    elif isinstance(clf, TabPFNClassifier):
        return load_surrogate_framework(DATABASE["tabpfn"])
    else:
        raise ValueError(
            "Is not possible to retrieve a surrogate framework based on the classifier type."
        )
    

def load_surrogate_framework(path: Path) -> Pipeline:
    return joblib.load(path)