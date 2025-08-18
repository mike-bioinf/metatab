import pandas as pd
from copy import deepcopy
from catboost import CatBoostClassifier
from estimators.estimators.params import TuningParams, DefaultParams
from estimators.estimators.base_gbdt import GBDTBaseEstimator



class MyCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of library default catboost without early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=DefaultParams.CATBOOST_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[_adjust_catboost_loss_function],
            n_threads_parameter="thread_count",
            early_stopping=False
        )



class MyESCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of the library default catboost classifier with early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=DefaultParams.ES_CATBOOST_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[
                _adjust_catboost_loss_function, 
                _adjust_es_catboost_logloss_metric
            ],
            n_threads_parameter="thread_count",
            early_stopping=True
        )



class MyTunedCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of tuned catboost without early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=TuningParams.CATBOOST_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[_adjust_catboost_loss_function],
            n_threads_parameter="thread_count",
            early_stopping=False
        )



class MyTunedESCatBoostClassifier(GBDTBaseEstimator):
    '''Implementation of tuned catboost with early stop'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=TuningParams.ES_CATBOOST_FIXED_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=CatBoostClassifier,
            callbacks_on_fixed_params=[
                _adjust_catboost_loss_function, 
                _adjust_es_catboost_logloss_metric
            ],
            n_threads_parameter="thread_count",
            early_stopping=True
        )



def _adjust_catboost_loss_function(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Set the loss function in the dict of parameters for the catboost classifier
    according to the classification scenario (binary or multi).
    Returns the modified/new dict.
    '''
    params = deepcopy(params) if copy else params
    n_classes = y.unique().size

    if n_classes == 2:
        params["loss_function"] = "Logloss"
        params["target_border"] = 0.5
    else:
        params["loss_function"] = "MultiClass"
        params["classes_count"] = n_classes

    return params



def _adjust_es_catboost_logloss_metric(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Adjust the logloss validation metric in the dict of params.
    Returns the modified/new dict.
    '''
    params = deepcopy(params) if copy else params

    if params["eval_metric"] != "logloss_to_adjust":
        return params
    
    n_classes = y.unique().size

    if n_classes == 2:
        params["eval_metric"] = "Logloss"
    else:
        params["eval_metric"] = "MultiClass"

    return params