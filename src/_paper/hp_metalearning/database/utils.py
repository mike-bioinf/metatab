import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from estimators.constants import Classifier



database = {
    "random_forest": Path("surrogate_pipeline_for_rf.joblib").resolve(),
    "xgb": Path("surrogate_pipeline_for_xgboost.joblib").resolve(),
    "catboost": Path("surrogate_pipeline_for_catboost.joblib").resolve(),
    "tabpfn": Path("surrogate_pipeline_for_tabpfn.joblib").resolve()
}


def query_surrogate_pipeline(clf_or_pipe: Classifier | Pipeline) -> Pipeline:
    '''
    Retrieve the fitted surrogate pipeline corresponding to a classifier.

    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Either a classifier instance or a pipeline whose last step is a Classifier.

    Returns:
        Pipeline: 
        The surrogate pipeline (column transformer + surrogate model) for the given classifier.
    '''
    clf = clf_or_pipe[-1] if isinstance(clf_or_pipe, Pipeline) else clf_or_pipe

    if isinstance(clf, RandomForestClassifier):
        return load_surrogate_pipeline(database["random_forest"])
    elif isinstance(clf, XGBClassifier):
        return load_surrogate_pipeline(database["xgb"])
    elif isinstance(clf, CatBoostClassifier):
        return load_surrogate_pipeline(database["catboost"])
    elif isinstance(clf, TabPFNClassifier):
        return load_surrogate_pipeline(database["tabpfn"])
    else:
        raise ValueError(
            "Is not possible to retrieve a surrogate model based on the type of classifier."
        )
    

def load_surrogate_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)