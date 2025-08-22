import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from estimators.params import TuningParams, DefaultParams

from estimators.base_gbdt import (
    GBDTBaseEstimator, 
    adjust_objective_logloss_and_num_classes,
    adjust_es_logloss_metric
)



## We have to use this decorator on predict methods of all LGBM classes and 
## on the fit of the tuned classes since prediction is performed in cross validation
def ignore_lgbm_feature_name_warning(method):
    '''
    Method decorator to filter the warning "X does not have valid feature names"
    raising from a bug in lgbm that checks at predict level the learned 
    artifical column names that it gives to numpy arrays at fit level.
    github issue: "https://github.com/microsoft/LightGBM/issues/6798".
    '''
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "X does not have valid feature names.*")
            return method(*args, **kwargs)
    return wrapper



class MyLGBMClassifier(GBDTBaseEstimator):
    '''Class that implements/wraps library LGBMClassifier.'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=DefaultParams.LGBM_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=LGBMClassifier,
            callbacks_on_fixed_params=[_adjust_lgbm_objective_and_num_classes],
            n_threads_parameter="n_jobs",
            early_stopping=False
        )

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X):
        return super().predict_proba(X)



class MyESLGBMClassifier(GBDTBaseEstimator):
    '''
    Class that wraps the library LGBMClassifier used 
    with early stopping on a validation set and without HPs tuning.
    '''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params=DefaultParams.ES_LGBM_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=LGBMClassifier,
            callbacks_on_fixed_params=[
                _adjust_lgbm_objective_and_num_classes, 
                _adjust_eslgbm_logloss_metric
            ],
            n_threads_parameter="n_jobs",
            early_stopping=True
        )

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X):
        return super().predict_proba(X)



class MyTunedLGBMClassifier(GBDTBaseEstimator):
    '''Class that implements tuned LGBMclassifier without early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params=TuningParams.LGBM_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=LGBMClassifier,
            callbacks_on_fixed_params=[_adjust_lgbm_objective_and_num_classes],
            n_threads_parameter="n_jobs",
            early_stopping=False
        )
    
    @ignore_lgbm_feature_name_warning
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X):
        return super().predict_proba(X)



class MyTunedESLGBMClassifier(GBDTBaseEstimator):
    '''Class that implements tuned LGBMClassifier with early stop on a validation set'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params = TuningParams.ES_LGBM_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=LGBMClassifier,
            callbacks_on_fixed_params=[
                _adjust_lgbm_objective_and_num_classes, 
                _adjust_eslgbm_logloss_metric
            ],
            n_threads_parameter="n_jobs",
            early_stopping=True
        )

    @ignore_lgbm_feature_name_warning
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X):
        return super().predict_proba(X)



def _adjust_lgbm_objective_and_num_classes(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Set the objective parameter to the params dict according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    return adjust_objective_logloss_and_num_classes(params, y, "lightgbm", copy)



def _adjust_eslgbm_logloss_metric(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Adjust the early stop logloss metric according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    return adjust_es_logloss_metric(params, y, "lightgbm", copy)