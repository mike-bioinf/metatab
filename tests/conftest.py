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
from metatab.estimators.params.space import TuningParams

from metatab.estimators.core.configurations import (
    EarlyStopConfiguration, 
    TuneConfiguration,
    EnsembleConfiguration
)

from metatab.estimators.estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyEnsembledRandomForestClassifier,
    MyExtraTreesClassifier,
    MyTunedExtraTreesClassifier,
    MyEnsembledExtraTreesClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier, 
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier,
    MyRealMLPClassifier,
    MyTunedRealMLPClassifier,
    MyEnsembledRealMLPClassifier,
    MyTabMClassifier,
    MyTunedTabMClassifier,
    MyEnsembledTabMClassifier
)

if TYPE_CHECKING:
    import pandas as pd
    from metatab.estimators.estimators import Estimator




### We define different parameters configuration to speed up the fitting procedure ----------------------------------

TEST_TUNE_CONFIGURATION = TuneConfiguration(
    algo="random",
    n_iter=5,
    n_cv_repeats=1,
    n_cv_folds=2,
    meta_strategy="best",
    params_distributions="" ## will be overwritten
)


TEST_ENSEMBLE_CONFIGURATION = EnsembleConfiguration(
    name="test",
    algo="random",
    n_members=1,
    save_path="", ## will be overwritten
    params_distributions="", ## will be overwritten
    raise_error_void_ensemble=False,
    log=50
)


TEST_RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 3
}

TEST_EXTRA_TREES_FIXED_PARAMS = {
    "n_estimators": 3
}

TEST_XGB_FIXED_PARAMS = {
    "n_estimators": 3,
    "verbosity": 0
}

TEST_ESXGB_FIXED_PARAMS = {
    "n_estimators": 3,
    "eval_metric": "logloss_to_adjust",
    "verbose_eval": False,
    "verbosity": 0
}

TEST_CATBOOST_FIXED_PARAMS = {
    "n_estimators": 3,
    "verbose": False,
    "allow_writing_files": False
}

TEST_ESCATBOOST_FIXED_PARAMS = {
    "n_estimators": 3,
    "eval_metric": "logloss_to_adjust",
    "verbose": False,
    "allow_writing_files": False
}

TEST_LGBM_FIXED_PARAMS = {
    "n_estimators": 3,
    "min_child_samples": 1,
    "verbose": -1
}

TEST_ESLGBM_FIXED_PARAMS = {
    "n_estimators": 3,
    "metric": "logloss_to_adjust",
    "min_child_samples": 1,
    "verbose": -1
}

TEST_TABPFN_FIXED_PARAMS = {
    "ignore_pretraining_limits": True
}

TEST_REALMLP_FIXED_PARAMS = {
    "n_epochs": 1,
    "n_ens": 1
}

TEST_TABM_FIXED_PARAMS = {
    "n_blocks": 1,
    "patience": 1
}


### Function to fit the estimators on the iris dataset --------------------------------------------------------------
def _fit_estimator(
    *,
    estimator: Estimator,
    fixed_params: dict | None,
    tune_configuration: TuneConfiguration | None,
    ensemble_configuration: EnsembleConfiguration | None,
    params_distributions: dict | None,
    file: str | Path, 
    X: pd.DataFrame, 
    y: pd.Series
):
    '''Fit the estimator on Xy and save the fitted model with pickle'''
    file = Path(file) if isinstance(file, str) else file
    fixed_params = {} if fixed_params is None else fixed_params

    if tune_configuration:
        tune_configuration = deepcopy(tune_configuration)
        tune_configuration.params_distributions = params_distributions

    if ensemble_configuration:
        ensemble_configuration = deepcopy(ensemble_configuration)
        ensemble_configuration.params_distributions = params_distributions
    
    estimator = estimator(
        preprocessing="base",
        seed=0,
        n_threads=4,
        device="auto",
        early_stop_configuration=EarlyStopConfiguration(),
        tune_configuration=tune_configuration,
        ensemble_configuration=ensemble_configuration
    )

    # overwriting fixed_params class attribute
    estimator.fixed_params = fixed_params
    estimator.fit(X, y).save(file)


X, y = load_iris(return_X_y=True, as_frame=True)
_fit_estimator_on_iris = partial(_fit_estimator, X=X, y=y)




### Configurations to test + fixture -----------------------------------------------------------------------
ESTIMATOR_DEFAULT_CONFIGS = {
    "my_rf_classifier.pkl": (MyRandomForestClassifier, TEST_RANDOM_FOREST_FIXED_PARAMS, None, None, None),
    "my_extra_trees_classifier.pkl": (MyExtraTreesClassifier, TEST_EXTRA_TREES_FIXED_PARAMS, None, None, None),
    "my_xgb_classifier.pkl": (MyXGBClassifier, TEST_XGB_FIXED_PARAMS, None, None, None),
    "my_es_xgb_classifier.pkl": (MyESXGBClassifier, TEST_ESXGB_FIXED_PARAMS, None, None, None),
    "my_catboost_classifier.pkl": (MyCatBoostClassifier, TEST_CATBOOST_FIXED_PARAMS, None, None, None),
    "my_es_catboost_classifier.pkl": (MyESCatBoostClassifier, TEST_ESCATBOOST_FIXED_PARAMS, None, None, None),
    "my_lgbm_classifier.pkl": (MyLGBMClassifier, TEST_LGBM_FIXED_PARAMS, None, None, None),
    "my_es_lgbm_classifier.pkl": (MyESLGBMClassifier, TEST_ESLGBM_FIXED_PARAMS, None, None, None),
    "my_tabpfn_classifier.pkl": (MyTabPFNClassifier, TEST_TABPFN_FIXED_PARAMS, None, None, None),
    "my_realmpl_classifier.pkl": (MyRealMLPClassifier, TEST_REALMLP_FIXED_PARAMS, None, None, None),
    "my_tabm_classifier.pkl": (MyTabMClassifier, TEST_TABM_FIXED_PARAMS, None, None, None),
}


ESTIMATOR_TUNE_CONFIGS = {
    "my_tuned_rf_classifier.pkl": (MyTunedRandomForestClassifier, TEST_RANDOM_FOREST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.RF),
    "my_extra_trees_classifier.pkl": (MyTunedExtraTreesClassifier, TEST_EXTRA_TREES_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.EXTRA_TREES),
    "my_tuned_xgb_classifier.pkl": (MyTunedXGBClassifier, TEST_XGB_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.XGB),
    "my_tuned_es_xgb_classifier.pkl": (MyTunedESXGBClassifier, TEST_ESXGB_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.XGB),
    "my_tuned_catboost_classifier.pkl": (MyTunedCatBoostClassifier, TEST_CATBOOST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.CATBOOST),
    "my_tuned_es_catboost_classifier.pkl": (MyTunedESCatBoostClassifier, TEST_ESCATBOOST_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.CATBOOST),
    "my_tuned_lgbm_classifier.pkl": (MyTunedLGBMClassifier, TEST_LGBM_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.LGMB),
    "my_tuned_es_lgbm_classifier.pkl": (MyTunedESLGBMClassifier, TEST_ESLGBM_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.LGMB),
    "my_tuned_tabpfn_classifier.pkl": (MyTunedTabPFNClassifier, TEST_TABPFN_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.TABPFN),
    "my_tuned_realmlp_classifier.pkl": (MyTunedRealMLPClassifier, TEST_REALMLP_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.REALMLP),
    "my_tuned_tabm_classifier.pkl": (MyTunedTabMClassifier, TEST_TABM_FIXED_PARAMS, TEST_TUNE_CONFIGURATION, None, TuningParams.TABM)
}


ESTIMATOR_ENSEMBLE_CONFIGS = {
    "my_ensembled_rf_classifier.pkl": (MyEnsembledRandomForestClassifier, TEST_RANDOM_FOREST_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.RF),
    "my_ensembled_extra_trees_classifier.pkl": (MyEnsembledExtraTreesClassifier, TEST_EXTRA_TREES_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.EXTRA_TREES),
    "my_ensembled_xgb_classifier.pkl": (MyEnsembledXGBClassifier, TEST_XGB_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.XGB),
    "my_ensembled_es_xgb_classifier.pkl": (MyEnsembledESXGBClassifier, TEST_ESXGB_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.XGB),
    "my_ensembled_catboost_classifier.pkl": (MyEnsembledCatBoostClassifier, TEST_CATBOOST_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.CATBOOST),
    "my_ensembled_es_catboost_classifier.pkl": (MyEnsembledESCatBoostClassifier, TEST_ESCATBOOST_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.CATBOOST),
    "my_ensembled_lgbm_classifier.pkl": (MyEnsembledLGBMClassifier, TEST_LGBM_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.LGMB),
    "my_ensembled_es_lgbm_classifier.pkl": (MyEnsembledESLGBMClassifier, TEST_ESLGBM_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.LGMB),
    "my_ensembled_tabpfn_classifier.pkl": (MyEnsembledTabPFNClassifier, TEST_TABPFN_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.TABPFN),
    "my_ensembled_realmlp_classifier.pkl": (MyEnsembledRealMLPClassifier, TEST_REALMLP_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.REALMLP),
    "my_ensembled_tabm_classifier.pkl": (MyEnsembledTabMClassifier, TEST_TABM_FIXED_PARAMS, None, TEST_ENSEMBLE_CONFIGURATION, TuningParams.TABM)
}


ESTIMATOR_ALL_CONFIGS = {
    **ESTIMATOR_DEFAULT_CONFIGS,
    **ESTIMATOR_TUNE_CONFIGS,
    **ESTIMATOR_ENSEMBLE_CONFIGS
}


@pytest.fixture(scope="session")
def fit_estimators_on_iris(tmp_path_factory) -> Path:
    '''
    Fit all estimators configs on the iris dataset and save them in a tmp folder.
    Returns the tmp folder.
    '''
    tmp_estimators_folder = tmp_path_factory.mktemp("estimators")

    for filename, (cls, fixed_params, tune_conf, ensemble_conf, tune_space) in ESTIMATOR_ALL_CONFIGS.items():
        if ensemble_conf:
            name_ensemble = re.sub("\\.pkl", "", filename)
            ensemble_conf.save_path = tmp_path_factory.mktemp(name_ensemble)

        _fit_estimator_on_iris(
            estimator=cls,
            fixed_params=fixed_params,
            tune_configuration=tune_conf,
            ensemble_configuration=ensemble_conf,
            params_distributions=tune_space,
            file=tmp_estimators_folder / filename,
        )

    return tmp_estimators_folder