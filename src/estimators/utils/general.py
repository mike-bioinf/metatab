from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Any, TYPE_CHECKING, Literal
from copy import deepcopy
from numpy.random import RandomState
from estimators.params.utils import DEFAULT_ESTIMATORS_TUNE_SPACES
from estimators.utils.constants import EARLY_STOPPED_ESTIMATORS

if TYPE_CHECKING:
    from metatab_utils.types import XType, YType
    from estimators.utils.types import EstimatorType
    from preprocessing.types import PreprocessingStrategy



def get_fresh_random_state(random_state: None | int | RandomState) -> RandomState:
    '''
    Get a fresh random state instance.
    If the input is None it generates a new instance seeded with 0.
    If int then it produces the random state using it as seed.
    If RandomState it returns a deepcopy of it.
    '''
    if random_state is None:
        return RandomState(0)
    elif isinstance(random_state, int):
        return RandomState(random_state)
    elif isinstance(random_state, RandomState):
        return deepcopy(random_state)
    else:
        raise ValueError("Unsupported input.")



def add_string_to_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to add at the beginning of dict keys a string.
    This is helpful when using sklearn pipelines.
    Notes that the function assumes that the keys are str.
    Returns a new dict.
    '''
    return {f"{string}{k}":v for k, v in params_dict.items()}



def remove_string_from_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to remove the string from the beginning of params dict keys.
    Notes that the function assumes that the keys are of str type.
    Returns a new dict.
    '''
    new_params_dict = {}
    for key, value in params_dict.items():
        new_key = key.removeprefix(string)
        new_params_dict[new_key] = value
    return new_params_dict



def update_dict( 
    dictionary: dict, 
    name_key: str, 
    value: Any, 
    copy: bool = False
) -> dict:
    '''
    Update the dict or a deepcopy of it with the name_key:value couple.
    Returns the updated dict.
    '''
    dictionary = deepcopy(dictionary) if copy else dictionary
    dictionary[name_key] = value
    return dictionary


# TODO: find a better location for this function
def check_meta_tuning_options(
    estimator: EstimatorType, 
    preprocessing: PreprocessingStrategy, 
    tune_space: str
) -> None:
    '''
    General check on meta-tuning related options:
    - checks that the meta-tuning option is requested with the right HPs space.
    - send a message when the preprocessing option is not suggested for meta-tuning. 
    '''
    estimator_default_space = DEFAULT_ESTIMATORS_TUNE_SPACES[estimator][0]

    if tune_space not in ["default", estimator_default_space]:
        raise ValueError(
            "'meta' algo can be used only with the estimator default space" + 
            f" ({estimator} --> {estimator_default_space})."
        )

    if (
        (estimator == "tabpfn" and preprocessing not in ["estimator_default", "density_filter"]) or
        (estimator != "tabpfn" and preprocessing not in ["estimator_default", "base"])    
    ):
        warnings.warn(
            "Metalearning is less effective when the following estimator-preprocessing couples are NOT respected:" +
            " tabpfn --> density_filter," +
            " others estimators --> base."
        )


# TODO: find a better location for thif function
def check_early_stop_options(
    estimator: EstimatorType,
    early_stop_rounds: int,
    validation_set_size: float
) -> None:
    '''
    General check on early stop related parameters. In detail checks:
    1. The estimator is a eraly stopped estimator.
    2. early_stop_rounds is positive (greater than 0)
    2. validation_set_size is a positive float in (0, 1].
    '''
    if estimator not in EARLY_STOPPED_ESTIMATORS:
        raise ValueError(f"'{estimator}' is not early stoppable.")
    if early_stop_rounds < 0:
        raise ValueError("'early_stop_rounds' must be a >= 0.")
    if not 0 < validation_set_size <=1:
        raise ValueError("'validation_set_size' must be a float in (0, 1].")



def collect_sklearn_classification_fit_info(
    obj: Any, 
    missing_feature_names_in: Literal["error", "none", "skip"] = "skip"
) -> dict:
    '''
    Utility to collect the classical sklearn classification 
    info `classes_`, `n_features_in_` and `feature_names_in_` 
    from a generic object. Here we assume that the attributes
    exists exept for `feature_names_in_`, which is optional.
    Th behaviour to assume when `feature_names_in_` is missing 
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



def collect_sklearn_classification_fit_info_from_data(X: XType, y: YType) -> dict:
    '''
    Collect the tipical sklearn classification info from the fit data.
    In detail we derive the `classes_`, `n_features_in_` and when
    possible the `feature_names_in_` info using these string as keys. 
    '''
    y = y.to_numpy() if isinstance(y, pd.Series) else y
    res = {"classes_": np.unique(y), "n_features_in_": X.shape[1]}

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



def check_y_is_integer_encoded(y: YType) -> None:
    '''Checks that y is integer encoded and dtyped'''
    y = y.to_numpy() if isinstance(y, pd.Series) else y
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Target variable y must be integer encoded and dtyped.")