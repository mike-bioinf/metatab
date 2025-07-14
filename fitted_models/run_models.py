from __future__  import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from sklearn.datasets import load_iris
from functools import partial
from estimators.types import Estimator

from estimators import (
    MyRandomizedXGBClassifier, 
    MyESRandomizedXGBClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyTabPFNClassifier
)

from .test_constants import (
    TEST_ES_XGBCLASSIFIER_FIXED_PARAMS,
    TEST_XGBCLASSIFIER_FIXED_PARAMS,
    TEST_RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS
)

if TYPE_CHECKING:
    import pandas as pd



def run_estimator(
    *,
    estimator: Estimator,
    fixed_params: dict | None,
    file: str | Path, 
    X: pd.DataFrame, 
    y: pd.Series
):
    '''
    Fit the estimator on Xy and save the fitted model with pickle. 
    If the model file already exists does nothing.
    '''
    file = Path(file) if isinstance(file, str) else file
    fixed_params = {} if fixed_params is None else fixed_params
    if not file.exists():
        estimator = estimator(preprocessing="base", seed=0, fixed_params=fixed_params)
        estimator.fit(X, y).save(file)


model_folder = Path(__file__).parents[0] / "fitted_models"
X, y = load_iris(return_X_y=True, as_frame=True)
run_estimator = partial(run_estimator, X=X, y=y)


run_estimator(
    estimator=MyESRandomizedXGBClassifier,
    fixed_params=TEST_ES_XGBCLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_es_randomized_xgb_classifier.pkl"
)

run_estimator(
    estimator=MyRandomizedXGBClassifier,
    fixed_params=TEST_XGBCLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_randomized_xgb_classifier.pkl"
)

run_estimator(
    estimator=MyXGBClassifier,
    fixed_params=TEST_XGBCLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_xgb_classifier.pkl"
)

run_estimator(
    estimator=MyESXGBClassifier,
    fixed_params=TEST_ES_XGBCLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_es_xgb_classifier.pkl"
)

run_estimator(
    estimator=MyRandomForestClassifier,
    fixed_params=TEST_RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_rf_classifier.pkl"
)

run_estimator(
    estimator=MyRandomizedRandomForestClassifier,
    fixed_params=TEST_RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS,
    file=model_folder / "my_randomized_rf_classifier.pkl"
)

run_estimator(
    estimator=MyTabPFNClassifier,
    fixed_params=None,
    file=model_folder / "my_tabpfn_classifier.pkl"
)