import pandas as pd
from xgboost import XGBClassifier
from estimators.params import TuningParams, DefaultParams

from estimators.base_gbdt import (
    GBDTBaseEstimator, 
    adjust_objective_logloss_and_num_classes,
    adjust_es_logloss_metric
)



class MyXGBClassifier(GBDTBaseEstimator):
    '''Class that implements/wraps library XGBClassifier.'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=DefaultParams.XGB_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[_adjust_xgb_objective_and_num_classes],
            n_threads_parameter="n_jobs",
            early_stopping=False
        )



class MyESXGBClassifier(GBDTBaseEstimator):
    '''
    Class that wraps the library XGBClassifier used 
    with early stopping on a validation set and without HPs tuning.
    '''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params=DefaultParams.ES_XGB_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[
                _adjust_xgb_objective_and_num_classes, 
                _adjust_esxgb_logloss_metric
            ],
            n_threads_parameter="n_jobs",
            early_stopping=True
        )



class MyTunedXGBClassifier(GBDTBaseEstimator):
    '''Class that implements Tuned xgboost without early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params=TuningParams.XGB_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[_adjust_xgb_objective_and_num_classes],
            n_threads_parameter="n_jobs",
            early_stopping=False
        )



class MyTunedESXGBClassifier(GBDTBaseEstimator):
    '''Class that implements tuned XGB with early stop on a validation set'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params = TuningParams.ES_XGB_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[
                _adjust_xgb_objective_and_num_classes, 
                _adjust_esxgb_logloss_metric
            ],
            n_threads_parameter="n_jobs",
            early_stopping=True
        )



def _adjust_xgb_objective_and_num_classes(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Set the objective parameter to the params dict according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    return adjust_objective_logloss_and_num_classes(params, y, "xgboost", copy)



def _adjust_esxgb_logloss_metric(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    The function adjust the logloss early stop metric according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    return adjust_es_logloss_metric(params, y, "xgboost", copy)