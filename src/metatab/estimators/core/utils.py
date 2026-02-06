"""
A set of functions useful to streamline the fit API.
They tend to do multiple things at once, so maybe to clean but functional.
"""

from __future__ import annotations

import re
import torch
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from sklearn.preprocessing import LabelEncoder
from metatab.preprocessing import create_classifier_pipeline
from metatab.metatab_utils.exceptions import DeviceError
from metatab.metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    import numpy as np
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.estimators.utils.types import EstimatorType
    from metatab.metatab_utils.types import XType, YType
    from sklearn.pipeline import Pipeline



def handle_device(
    device: Literal["cpu", "gpu", "auto"], 
    type_estimator: EstimatorType
) -> Literal["cpu", "gpu"]:
    '''
    Handle the device info at fit level.
    In details resolve the "auto" info depending on the estimator,
    and raise a DeviceError when the input device and estimator are incompatible.

    Returns:
        Literal["cpu","gpu"]: The resolved device.
    '''
    resolved_device = _resolve_device(device, type_estimator)
    if resolved_device == "cuda": _check_cuda_is_available()
    _check_device_estimator_combination(resolved_device, type_estimator)
    return resolved_device
    


def create_pipeline(
    classifier_class,
    type_estimator: EstimatorType,
    fixed_params: dict,
    callbacks_on_fixed_params,
    y: YType,
    preprocessing: PreprocessingStrategy,
    classifier_random_state_parameter: str | None,
    classifier_nthreads_paramater: str | None,
    classifier_device_parameter: str | None,
    seed: int,
    n_threads: int,
    resolved_device: Literal["cpu", "gpu"],
) -> Pipeline:
    '''
    Create the pipeline headed by the classsifier to fit. In details cares of:
    - finalizing the classifier fixed params with seed, n_threads, device info plus callbacks
    - creating and returning the pipeline

    Notes: it create always a deepcopy of the dixed params on which to operate.

    Returns:
        Pipeline: The pipeline headed by the classifier.
    '''
    params = deepcopy(fixed_params)

    if classifier_random_state_parameter: 
        params[classifier_random_state_parameter] = seed

    if classifier_nthreads_paramater: 
        params[classifier_nthreads_paramater] = n_threads

    if classifier_device_parameter:
        params[classifier_device_parameter] = resolved_device
    
    for callback in ensure_or_create(callbacks_on_fixed_params, list):
        params = callback(params, y, False)

    pipe = create_classifier_pipeline(
        preprocessing=preprocessing,
        density_feature_selector_strategy="exact", ###REFACTOR: CAPIRE COME GESTIRLO (FISSO?)
        classifier=classifier_class,
        classifier_params=fixed_params,
        type_estimator=type_estimator
    )

    return pipe



def encode_y(X: XType, y: YType) -> tuple[LabelEncoder, YType]:
    '''
    Encode the y using sklearn LabelEncoder.
    The X is needed only to uniform the resulting y type to X.
    Returns the sklearn label encoder and the encoded y. 
    '''
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity
    return le, y



def check_validation_set(validation_set: float | tuple[XType, YType] | None) -> None:   
    if isinstance(validation_set, float):
        if not 0 < validation_set < 1:
            raise ValueError("'validation_set' must be a float in (0, 1).")

    elif isinstance(validation_set, tuple):
        if len(validation_set) != 2:
            raise ValueError(
                "validation_set must be a tuple of (X, y)"
            )
        
        X, y = validation_set
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "X must be a pd.DataFrame or np.ndarray"
            )
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                "y must be a pd.Series or np.ndarray"
            )

        if X.shape[0] != len(y):
            raise ValueError(
                "validation_set shape mismatch: X rows != y length"
            )

    elif validation_set is None:
        pass

    else:
        raise TypeError(
            "validation_set must be None, a float, or a tuple (X, y)"
        )



def _resolve_device(device: Literal["cpu", "cuda", "auto"], estimator: EstimatorType) -> Literal["cpu", "cuda"]:
    '''
    Resolve the device based on the device and estimator inputs.
    The utility returns the device when not "auto", otherwise "cuda" 
    if available AND the estimator requires it, or "cpu".
    '''
    if device == "auto":
        if estimator in ["tabpfn", "realmlp", "tabm"]:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device
    return resolved_device



def _check_device_estimator_combination(device: Literal["cpu", "cuda", "auto"], estimator: EstimatorType) -> None:
    '''Checks whether the specified estimator can be run on the specified device'''
    # uniform "es" and "base" estimator versions
    estimator = re.sub("^es_", "", estimator)

    if estimator in ["random_forest", "extra_trees"] and device == "cuda":
        raise DeviceError(f"cuda is not compatible with {estimator} estimator.")
    elif estimator in ["catboost", "xgb", "lgbm"] and device == "cuda":
        raise DeviceError(f"Metatab does not support {estimator} on GPU.")
    


def _check_cuda_is_available() -> None:
    '''Checks whether cuda is available raising an error otherwise'''
    if not torch.cuda.is_available():
        raise DeviceError("cuda is requested but not available.")