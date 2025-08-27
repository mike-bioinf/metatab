import pandas as pd
from catboost import CatBoostClassifier
from estimators.params import TuningParams, DefaultParams

from estimators.base_gbdt import (
    GBDTBaseEstimator,
    adjust_es_logloss_metric,
    adjust_objective_logloss_and_num_classes
)



class MyCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of library default catboost classifier without early stop'''
    fixed_params=DefaultParams.CATBOOST_DEFAULT_PARAMS

    def __init__(
        self, preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration
    ):
        super().__init__(
            preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[_adjust_catboost_loss_function_and_num_classes],
            n_threads_parameter="thread_count",
            early_stopping=False
        )



class MyESCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of the library default catboost classifier with early stop'''
    fixed_params=DefaultParams.ES_CATBOOST_DEFAULT_PARAMS

    def __init__(
        self, preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration
    ):
        super().__init__(
            preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[
                _adjust_catboost_loss_function_and_num_classes, 
                _adjust_es_catboost_logloss_metric
            ],
            n_threads_parameter="thread_count",
            early_stopping=True
        )



class MyTunedCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of tuned catboost without early stop'''
    fixed_params=TuningParams.CATBOOST_FIXED_PARAMS

    def __init__(
        self, preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration
    ):
        super().__init__(
            preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[_adjust_catboost_loss_function_and_num_classes],
            n_threads_parameter="thread_count",
            early_stopping=False
        )



class MyTunedESCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of tuned catboost with early stop'''
    fixed_params=TuningParams.ES_CATBOOST_FIXED_PARAMS

    def __init__(
        self, preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration
    ):
        super().__init__(
            preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[
                _adjust_catboost_loss_function_and_num_classes, 
                _adjust_es_catboost_logloss_metric
            ],
            n_threads_parameter="thread_count",
            early_stopping=True
        )



def _adjust_catboost_loss_function_and_num_classes(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Set the classification logloss objective according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    return adjust_objective_logloss_and_num_classes(params, y, "catboost", copy)



def _adjust_es_catboost_logloss_metric(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Adjust the logloss validation metric according to the classification scenario.
    Returns the modified/new dict.
    '''
    return adjust_es_logloss_metric(params, y, "catboost", copy)