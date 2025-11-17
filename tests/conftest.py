'''
Configuration file to fit all estimators on the iris dataset
'''

from __future__  import annotations

import pytest
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from sklearn.datasets import load_iris
from functools import partial
from estimators.params import TuningParams

from estimators.estimators import (
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
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
)

if TYPE_CHECKING:
    import pandas as pd
    from estimators.estimators import Estimator




### We define different parameters configuration to speed up the fitting procedure ----------------------------------

TEST_TUNE_CONFIGURATION = {
    "configuration": "c0",
    "algo": "random",
    "n_iter": 5,
    "n_repeats": 1,
    "n_folds": 3,
    "meta_strategy": "best",
    "meta_strategy_params": None,
    "meta_surrogate_model": None
}

TEST_RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 10
}

TEST_XGB_FIXED_PARAMS = {
    "n_estimators": 10,
    "verbosity": 0
}

TEST_ESXGB_FIXED_PARAMS = {
    "n_estimators": 10,
    "eval_metric": "logloss_to_adjust",
    "verbose_eval": False,
    "verbosity": 0
}

TEST_CATBOOST_FIXED_PARAMS = {
    "n_estimators": 10,
    "verbose": False,
    "allow_writing_files": False
}

TEST_ESCATBOOST_FIXED_PARAMS = {
    "n_estimators": 10,
    "eval_metric": "logloss_to_adjust",
    "verbose": False,
    "allow_writing_files": False
}

TEST_LGBM_FIXED_PARAMS = {
    "n_estimators": 10,
    "min_child_samples": 1,
    "verbose": -1
}

TEST_ESLGBM_FIXED_PARAMS = {
    "n_estimators": 10,
    "metric": "logloss_to_adjust",
    "min_child_samples": 1,
    "verbose": -1
}

TEST_TABPFN_FIXED_PARAMS = {
    "ignore_pretraining_limits": True
}

# TEST_FINETUNED_TABPFN_FIXED_PARAMS = {
#     "finetune_setup": {"max_steps": 2},
#     "tabpfn_classifier_params": {"ignore_pretraining_limits": True},
#     "n_accumulation_steps": 1,
#     "log": False
# }




### Function to fit the estimators on the iris dataset --------------------------------------------------------------
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
        preprocessing="estimator_default", 
        seed=0,
        n_threads=4,
        early_stop_configuration={
            "early_stop_rounds": 4, 
            "validation_set_size": 0.3
        },
        tune_configuration=tune_configuration
    )

    # overwriting fixed_params class attribute
    estimator.fixed_params = fixed_params
    estimator.fit(X, y).save(file)


X, y = load_iris(return_X_y=True, as_frame=True)
_fit_estimator_on_iris = partial(_fit_estimator, X=X, y=y)




### Configurations to test + fixture -----------------------------------------------------------------------
ESTIMATOR_DEFAULT_CONFIGS = {
    "my_rf_classifier.pkl": (MyRandomForestClassifier, TEST_RANDOM_FOREST_FIXED_PARAMS, None, None),
    "my_xgb_classifier.pkl": (MyXGBClassifier, TEST_XGB_FIXED_PARAMS, None, None),
    "my_es_xgb_classifier.pkl": (MyESXGBClassifier, TEST_ESXGB_FIXED_PARAMS, None, None),
    "my_catboost_classifier.pkl": (MyCatBoostClassifier, TEST_CATBOOST_FIXED_PARAMS, None, None),
    "my_es_catboost_classifier.pkl": (MyESCatBoostClassifier, TEST_ESCATBOOST_FIXED_PARAMS, None, None),
    "my_lgbm_classifier.pkl": (MyLGBMClassifier, TEST_LGBM_FIXED_PARAMS, None, None),
    "my_es_lgbm_classifier.pkl": (MyESLGBMClassifier, TEST_ESLGBM_FIXED_PARAMS, None, None),
    "my_tabpfn_classifier.pkl": (MyTabPFNClassifier, TEST_TABPFN_FIXED_PARAMS, None, None),
    #"my_aesfinetunedtabpfn_classifier.pkl": (MyAesFineTunedTabPFNClassifier, TEST_FINETUNED_TABPFN_FIXED_PARAMS, None, None),
}


ESTIMATOR_TUNE_CONFIGS = {
    "my_tuned_rf_classifier.pkl": (MyTunedRandomForestClassifier, TEST_RANDOM_FOREST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.RF_C0),
    "my_tuned_xgb_classifier.pkl": (MyTunedXGBClassifier, TEST_XGB_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.XGB_C0),
    "my_tuned_es_xgb_classifier.pkl": (MyTunedESXGBClassifier, TEST_ESXGB_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.XGB_C0),
    "my_tuned_catboost_classifier.pkl": (MyTunedCatBoostClassifier, TEST_CATBOOST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.CATBOOST_C0),
    "my_tuned_es_catboost_classifier.pkl": (MyTunedESCatBoostClassifier, TEST_ESCATBOOST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.CATBOOST_C0),
    "my_tuned_lgbm_classifier.pkl": (MyTunedLGBMClassifier, TEST_LGBM_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.LGMB_C0),
    "my_tuned_es_lgbm_classifier.pkl": (MyTunedESLGBMClassifier, TEST_ESLGBM_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.LGMB_C0),
    "my_tuned_tabpfn_classifier.pkl": (MyTunedTabPFNClassifier, TEST_TABPFN_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, TuningParams.TABPFN_C0),
}


ESTIMATOR_ALL_CONFIGS = {
    **ESTIMATOR_DEFAULT_CONFIGS,
    **ESTIMATOR_TUNE_CONFIGS
}


@pytest.fixture(scope="session")
def fit_estimators_on_iris(tmp_path_factory) -> Path:
    '''
    Fit all estimators configs on the iris dataset and save them in a tmp folder.
    Returns the tmp folder.
    '''
    tmp_estimators_folder = tmp_path_factory.mktemp("estimators")

    for filename, (cls, fixed_params, tune_configuration, tune_space) in ESTIMATOR_ALL_CONFIGS.items():
        _fit_estimator_on_iris(
            estimator=cls,
            fixed_params=fixed_params,
            tune_configuration=tune_configuration,
            params_distributions=tune_space,
            file=tmp_estimators_folder / filename,
        )

    return tmp_estimators_folder




### Alternative tune space configurations + fixture ----------------------------------------------------------------------------- 
ESTIMATOR_ALTERNATIVE_TUNE_CONFIGS = {
    "xgb": (MyTunedXGBClassifier, TuningParams.XGB_FIXED_PARAMS, ["c1", "c2", "c3", "c4"], "XGB"),
    "es_xgb": (MyTunedESXGBClassifier, TuningParams.ES_XGB_FIXED_PARAMS, ["c1", "c2", "c3", "c4"], "XGB"),
    "catboost": (MyTunedCatBoostClassifier, TuningParams.CATBOOST_FIXED_PARAMS, ["c1", "c2", "c3", "c4", "c5"], "CATBOOST"),
    "es_catboost": (MyTunedESCatBoostClassifier, TuningParams.ES_CATBOOST_FIXED_PARAMS, ["c1", "c2", "c3", "c4", "c5"], "CATBOOST")
}


def _fit_alternative_tune_spaces(
    *,
    cls,
    fixed_params,
    spaces: list[str],
    tuning_param_prefix: str,
    basename_models: str,
    folder: Path
) -> None:
    '''
    Generic helper to fit tunable estimators with multiple tune spaces.

    Parameters:
        cls: estimator class.
        fixed_params: estimator fixed params.
        spaces (list[str]): list of tuning spaces labels (i.e. "c1").
        tuning_param_prefix (str): Prefix of the tune spaces (i.e. "XGB" from "XGB_C1").
        basename_models (str): Basename of the saved model files.
        folder (Path): Folder where the models are saved. 
    '''
    for space in spaces:
        conf = deepcopy(TEST_TUNE_CONFIGURATION)
        conf["configuration"] = space
        params = getattr(TuningParams, f"{tuning_param_prefix}_C{re.sub("c", "", space)}")
        _fit_estimator_on_iris(
            estimator=cls,
            fixed_params=fixed_params,
            tune_configuration=conf,
            params_distributions=params,
            file=folder / f"{basename_models}_{space}.pkl",
        )


@pytest.fixture(scope="session")
def fit_estimators_alternative_tune_configs(tmp_path_factory) -> Path:
    '''
    Fit the estimator alternative tune configurations in the same tmp folder.
    Returns the tmp folder as a Path object.
    '''
    tmp_folder = tmp_path_factory.mktemp("estimators_tune_alternative")
    for estimator, (cls, fixed_params, spaces, tuning_param_prefix) in ESTIMATOR_ALTERNATIVE_TUNE_CONFIGS.items():
        _fit_alternative_tune_spaces(
            cls=cls,
            fixed_params=fixed_params,
            spaces=spaces,
            tuning_param_prefix=tuning_param_prefix,
            basename_models=estimator,
            folder=tmp_folder
        )
    return tmp_folder


def get_alternative_estimator_file_names() -> list[str]:
    '''Helper to get automatically the alternative tuned estimators file names'''
    names = []
    for estimator, (_, _, spaces, *_) in ESTIMATOR_ALTERNATIVE_TUNE_CONFIGS.items():
        names.extend([f"{estimator}_{space}.pkl" for space in spaces])
    return names
