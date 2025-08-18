import pandas as pd
from copy import deepcopy
from xgboost import XGBClassifier
from estimators.estimators.params import TuningParams, DefaultParams
from estimators.estimators.base_gbdt import GBDTBaseEstimator



class MyXGBClassifier(GBDTBaseEstimator):
    '''Class that implements/wraps library XGBClassifier.'''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration, 
        fixed_params=DefaultParams.XGB_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[_adjust_xgb_objective],
            n_threads_parameter="n_jobs",
            early_stopping=False
        )



class MyESXGBClassifier(GBDTBaseEstimator):
    '''
    Class that wraps the XGBClassifier used with early stopping
    on a validation set and without HPs tuning.
    '''
    def __init__(
        self, preprocessing, seed, n_threads, tune_configuration,
        fixed_params=DefaultParams.ES_XGB_DEFAULT_PARAMS
    ):
        super().__init__(
            preprocessing, seed, n_threads, tune_configuration, fixed_params,
            classifier_cls=XGBClassifier,
            callbacks_on_fixed_params=[
                _adjust_xgb_objective, 
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
            callbacks_on_fixed_params=[_adjust_xgb_objective],
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
                _adjust_xgb_objective, 
                _adjust_esxgb_logloss_metric
            ],
            n_threads_parameter="n_jobs",
            early_stopping=True
        )




def _adjust_xgb_objective(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Add the objective parameter since it is fit/data specific.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    params = deepcopy(params) if copy else params
    n_classes = y.unique().size
    
    if n_classes == 2:
        # here we must NOT specify num_class otherwise strange behaviours
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    
    return params 



def _adjust_esxgb_logloss_metric(params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    The XGBboost package differentiate between binary "logloss"
    and multiclassification "mlogloss" for the early stopping metric.
    The function adjust the logloss according to the classification scenario.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    params = deepcopy(params) if copy else params

    if params["eval_metric"] != "logloss_to_adjust":
        return params
    
    n_classes = y.unique().size
    
    if n_classes == 2:
        params["eval_metric"] = "logloss"
    else:
        params["eval_metric"] = "mlogloss"
    
    return params