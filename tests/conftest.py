'''
Configuration file to fit all estimators on the iris dataset
'''

from __future__  import annotations

import pytest
from pathlib import Path
from typing import TYPE_CHECKING
from sklearn.datasets import load_iris
from functools import partial
from estimators import Estimator
from estimators.params import TuningParams

from estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier, 
    MyTunedESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyTabPFNClassifier
)

if TYPE_CHECKING:
    import pandas as pd



def _fit_estimator(
    *,
    estimator: Estimator,
    fixed_params: dict | None,
    tune_configuration: dict | None,
    params_distributions: dict | None,
    file: str | Path, 
    X: pd.DataFrame, 
    y: pd.Series
):
    '''Fit the estimator on Xy and save the fitted model with pickle'''
    file = Path(file) if isinstance(file, str) else file
    fixed_params = {} if fixed_params is None else fixed_params

    if tune_configuration:
        tune_configuration["params_distributions"] = params_distributions
    
    estimator = estimator(
        preprocessing="base", 
        seed=0,
        n_threads=4,
        tune_configuration=tune_configuration,
        fixed_params=fixed_params
    )

    estimator.fit(X, y).save(file)


X, y = load_iris(return_X_y=True, as_frame=True)
_fit_estimator_on_iris = partial(_fit_estimator, X=X, y=y)



# Here we define different sets of model parameters 
# to speed up the fitting procedure.

TEST_TUNE_CONFIGURATION = {
    "configuration": "c0",
    "algo": "tpe",
    "n_iter": 2,
    "n_repeats": 1,
    "n_splits": 5
}

TEST_RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 10
}

TEST_XGB_FIXED_PARAMS = {
    "n_estimators": 10,
    "verbosity": 0
}

TEST_ES_XGB_FIXED_PARAMS = {
    "n_estimators": 10,
    "eval_metric": "logloss_to_adjust",
    "early_stopping_rounds": 4,
    "verbose_eval": False,
    "verbosity": 0
}

TEST_CATBOOST_FIXED_PARAMS = {
    "n_estimators": 10,
    "verbose": False,
    "allow_writing_files": False
}

TEST_ES_CATBOOST_FIXED_PARAMS = {
    "n_estimators": 10,
    "eval_metric": "logloss_to_adjust",
    "early_stopping_rounds": 4,
    "verbose": False,
    "allow_writing_files": False
}

TEST_LGBM_FIXED_PARAMS = {
    "n_estimators": 10,
    "min_child_samples": 1,
    "verbose": -1
}

TEST_ES_LGBM_FIXED_PARAMS = {
    "n_estimators": 10,
    "early_stopping_rounds": 4,
    "metric": "logloss_to_adjust",
    "min_child_samples": 1,
    "verbose": -1
}

TEST_TABPFN_FIXED_PARAMS = {
    "ignore_pretraining_limits": True,
    "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
}



@pytest.fixture(scope="session")
def fit_estimators_on_iris(tmp_path_factory) -> Path:
    '''
    Fit all estimators on the iris dataset saving them in a tmp folder.
    Return the tmp folder Path object to the test function.
    '''
    tmp_estimators_folder = tmp_path_factory.mktemp("estimators")
    
    _fit_estimator_on_iris(
        estimator=MyRandomForestClassifier,
        fixed_params=TEST_RANDOM_FOREST_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_rf_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedRandomForestClassifier,
        fixed_params=TEST_RANDOM_FOREST_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.RF_C0,
        file=tmp_estimators_folder / "my_tuned_rf_classifier.pkl"
    )
    
    _fit_estimator_on_iris(
        estimator=MyXGBClassifier,
        fixed_params=TEST_XGB_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_xgb_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyESXGBClassifier,
        fixed_params=TEST_ES_XGB_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_es_xgb_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedXGBClassifier,
        fixed_params=TEST_XGB_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.XGB_C0,
        file=tmp_estimators_folder / "my_tuned_xgb_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedESXGBClassifier,
        fixed_params=TEST_ES_XGB_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.XGB_C0,
        file=tmp_estimators_folder / "my_tuned_es_xgb_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyCatBoostClassifier,
        fixed_params=TEST_CATBOOST_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_catboost_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyESCatBoostClassifier,
        fixed_params=TEST_ES_CATBOOST_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_es_catboost_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedCatBoostClassifier,
        fixed_params=TEST_CATBOOST_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.CATBOOST_C0,
        file=tmp_estimators_folder / "my_tuned_catboost_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedESCatBoostClassifier,
        fixed_params=TEST_ES_CATBOOST_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.CATBOOST_C0,
        file=tmp_estimators_folder / "my_tuned_es_catboost_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyLGBMClassifier,
        fixed_params=TEST_LGBM_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_lgbm_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyESLGBMClassifier,
        fixed_params=TEST_ES_LGBM_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_es_lgbm_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedLGBMClassifier,
        fixed_params=TEST_LGBM_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.LGMB_C0,
        file=tmp_estimators_folder / "my_tuned_lgbm_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTunedESLGBMClassifier,
        fixed_params=TEST_ES_LGBM_FIXED_PARAMS,
        tune_configuration=TEST_TUNE_CONFIGURATION,
        params_distributions=TuningParams.LGMB_C0,
        file=tmp_estimators_folder / "my_tuned_es_lgbm_classifier.pkl"
    )

    _fit_estimator_on_iris(
        estimator=MyTabPFNClassifier,
        fixed_params=TEST_TABPFN_FIXED_PARAMS,
        tune_configuration=None,
        params_distributions=None,
        file=tmp_estimators_folder / "my_tabpfn_classifier.pkl"
    )

    return tmp_estimators_folder