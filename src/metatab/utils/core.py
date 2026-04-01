from __future__ import annotations

import time
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from metatab.utils.types import XType, YType, Classifier
    from sklearn.pipeline import Pipeline



def add_prefix_to_params_when_absent(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to add at the beginning of dict keys a string IF NOT already present.
    This is helpful when using sklearn pipelines.
    Note that the function assumes that the dict keys are str.
    Returns a new dict.
    '''
    return {
        f"{string}{k}": v 
        for k, v in params_dict.items()
        if not k.startswith(string)
    }



def remove_prefix_from_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to remove the string from the beginning of params dict keys.
    Note that the function assumes that the keys are of str type.
    Returns a new dict.
    '''
    new_params_dict = {}
    for key, value in params_dict.items():
        new_key = key.removeprefix(string)
        new_params_dict[new_key] = value
    return new_params_dict



def fit_using_validation_set(
    *,
    pipe: Pipeline,
    X: XType,
    y: YType,
    X_val : XType | None = None,
    y_val: YType | None = None,
    validation_set_size: float | None = None,
    seed: int,
    return_fit_time: bool = False
 ) -> Pipeline | tuple[Pipeline, float]:
    '''
    Utility to fit a classifier in a pipe needing a validation set.
    Supports both external validation or draws the set from the input data.
    The classifier (pipe head) must support the validation set at its fit interface, 
    through the "eval_set" parameter, accepting a tuple (X_val, y_val).

    Parameters:
        pipe (Pipeline): 
            The pipeline to fit. Is assumed to end with a classifier.
        
        X (XType): Training feature space.
        
        y (YType): Training labels.
        
        X_val (XType | None):
            X data to use as validation (external to X).

        y_val (YType | None):
            y data to use as validation (external to y)
        
        validation_set_size (float | None): 
            Ratio of training data to use as validation.
            The validation set is drawn from "X" and "y".
            Must be a number in (0, 1) or None.

        seed (int): 
            Seed for reproducibility used ONLY in the train/val splitting.
            Used only when "validation_set_size" is specified.
        
        return_fit_time (bool, optional):
            Whether to return the fit time along with the fitted pipe.
            If True returns a tuple [pipe, fit_time], otherwise pipe directly.

    Returns:
        Pipeline|tuple: 
        The fitted pipeline alone or in a tuple with the fit time.
    '''
    if X_val is None and y_val is not None:
        raise ValueError("X_val is None and y_val is not")

    if y_val is None and X_val is not None:
        raise ValueError("y_val is None and X_val is not.")

    if X_val is None and validation_set_size is None:
        raise ValueError("One between X/y_val and validation_set_size must be specified")

    if X_val is not None and validation_set_size is not None:
        raise ValueError("Both X/y_val and validation_set_size are specified. Pick one option.")

    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=validation_set_size,
            random_state=seed,
            stratify=y
        )
    else:
        X_train, y_train = X, y

    # we always consider the preprocessing in the fit time
    start_fit_time = time.time()

    if len(pipe) > 1:
        # we split the classifier from the preprocessing 
        # to avoid to repeat the preprocessing 2 times.
        # we fit in place the two components separately.
        clf: Classifier = pipe[-1]
        preprocessing_pipeline: Pipeline = pipe[:-1]
        X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
        X_val_transformed = preprocessing_pipeline.transform(X_val)
        clf.fit(X_train_transformed, y_train, eval_set=[(X_val_transformed, y_val)])
    else:
        # we fit directly the classifier
        pipe[-1].fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    fit_time = time.time() - start_fit_time

    if return_fit_time:
        return [pipe, fit_time]
    else:
        return pipe



# refactor: not used?
def collect_sklearn_classification_fit_info(
    obj: Any,
    missing_feature_names_in: Literal["error", "none", "skip"] = "skip"
) -> dict:
    '''
    Utility to collect the classical sklearn classification 
    info `classes_`, `n_features_in_` and `feature_names_in_` 
    from a generic object. Here we assume that the attributes
    exists exept for `feature_names_in_`, which is optional.
    The behaviour to assume when `feature_names_in_` is missing 
    is encoded through the `missing_features_names_in` parameter.
    '''
    res = {
        "classes_": obj.classes_,
        "n_features_in_": obj.n_features_in_
    }
    
    if not hasattr(obj, "feature_names_in_"):
        if missing_feature_names_in == "error":
            raise ValueError("obj does not have 'feature_names_in_' attribute.")
        elif missing_feature_names_in == "none":
            res["feature_names_in_"] = None
        elif missing_feature_names_in == "skip":
            pass
        else:
            raise ValueError(
                "'missing_features_names_in' must be equal to 'error', 'none' or 'skip'."
            )
    else:
        res["feature_names_in_"] = obj.feature_names_in_
    
    return res



def learn_sklearn_features_attributes(X: XType) -> dict:
    '''
    Learn the tipical sklearn classification info from the X fit data.
    In detail we derive `n_features_in_` and when possible the `feature_names_in_` 
    info using these string as keys. 
    '''
    res = {"n_features_in_": X.shape[1]}
    if isinstance(X, pd.DataFrame) and all([isinstance(col, str) for col in X.columns]):
        res["feature_names_in_"] = X.columns
    return res



def check_predict_features(obj: Any, X: XType) -> None:
    '''
    Check done on the X passed in predict_*/tranform methods for sklearn-like estimators 
    objects that learn `n_features_in_` and `feature_names_in_` attributes at fit level.
    '''
    n_features = X.shape[1]

    if n_features != obj.n_features_in_:
        raise ValueError(
            "Different number of features between fit" + 
            f" ({obj.n_features_in_}) and predict ({n_features}) calls."
        )

    if isinstance(X, pd.DataFrame) and hasattr(obj, "feature_names_in_"):
        if not all([isinstance(col, str) for col in X.columns]):
            raise ValueError("X has not all string columns.")
        if (X.columns != obj.feature_names_in_).any():
            raise ValueError("Different column names between fit and predict calls.")
        


def adjust_objective_logloss_and_num_classes(
    params: dict, 
    y: np.ndarray | pd.Series,
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
    y = y.to_numpy() if isinstance(y, pd.Series) else y
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
    y: np.ndarray | pd.Series, 
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
    
    # return the params dict when the early stop metric is not the adjustable logloss
    if params[metric_parameter] != "logloss_to_adjust":
        return params
    
    y = y.to_numpy() if isinstance(y, pd.Series) else y
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
    
    if n_classes == 2:
        params[metric_parameter] = binary_logloss
    else:
        params[metric_parameter] = multi_logloss
    
    return params