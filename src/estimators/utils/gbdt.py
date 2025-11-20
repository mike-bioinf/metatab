import numpy as np
from copy import deepcopy
from typing import Literal



def adjust_objective_logloss_and_num_classes(
    params: dict, 
    y: np.ndarray,
    copy: bool,
    framework: Literal["catboost", "xgboost", "lightgbm"]
) -> dict:
    '''
    GBDT classification framework implementations need to dinamically adjust the objective loss 
    and number of input classes depending on the classification scenario (binary or multi).
    In addiction different framework encode with different names the same losses functions.
    The function works on the input dict of params or a deepcopy of it (copy parameter).
    Returns the updated dict of params.
    '''
    params = deepcopy(params) if copy else params
    n_classes = np.unique(y).size
    
    if framework == "catboost":
        loss_parameter = "loss_function"
        binary_logloss = "Logloss"
        multi_logloss = "MultiClass"
        n_classes_parameter = "classes_count"
    elif framework == "xgboost":
        loss_parameter = "objective"
        binary_logloss = "binary:logistic"
        multi_logloss = "multi:softprob"
        n_classes_parameter = "num_class"
    elif framework == "lightgbm":
        loss_parameter = "objective"
        binary_logloss = "binary"
        multi_logloss = "multiclass"
        n_classes_parameter = "num_class"
    else:
        raise ValueError("Unsupported GBDT framework.")
    
    if n_classes == 2:
        params[loss_parameter] = binary_logloss
    else:
        params[loss_parameter] = multi_logloss
        params[n_classes_parameter] = n_classes

    return params



def adjust_es_logloss_metric(
    params: dict, 
    y: np.ndarray, 
    copy: bool,
    framework: Literal["catboost", "xgboost", "lightgbm"]
) -> dict:
    '''
    The GBDT classification implementations differentiate between 
    binary and multi logloss as early stopping metric.
    The function adjust the evaluation logloss metric marked as "logloss_to_adjust".
    The function works on the input dict of params or a deepcopy of it (copy parameter).
    Returns the updated dict of params.
    '''
    params = deepcopy(params) if copy else params
    n_classes = np.unique(y).size

    if framework == "catboost":
        metric_parameter = "eval_metric"
        binary_logloss = "Logloss"
        multi_logloss = "MultiClass"
    elif framework == "xgboost":
        metric_parameter = "eval_metric"
        binary_logloss = "logloss"
        multi_logloss = "mlogloss"
    elif framework == "lightgbm":
        metric_parameter = "metric"
        binary_logloss = "binary_logloss"
        multi_logloss = "multi_logloss"
    else:
        raise ValueError("Unsupported GBDT framework.")
    
    # return the params dict when the early stop metric is not the adjustable logloss
    if params[metric_parameter] != "logloss_to_adjust":
        return params
    
    if n_classes == 2:
        params[metric_parameter] = binary_logloss
    else:
        params[metric_parameter] = multi_logloss
    
    return params