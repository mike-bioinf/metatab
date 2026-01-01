from __future__ import annotations

import re
import torch
from typing_extensions import TYPE_CHECKING, Literal
from metatab.metatab_utils.exceptions import DeviceError

if TYPE_CHECKING:
    from metatab.estimators.utils.types import EstimatorType



def resolve_device(device: Literal["cpu", "cuda", "auto"], estimator: EstimatorType) -> Literal["cpu", "cuda"]:
    '''
    Resolve the device based on the device and estimator inputs.
    The utility returns the device when not "auto", otherwise "cuda" 
    if available AND the estimator requires it, or "cpu".
    '''
    if device == "auto":
        if estimator in ["tabpfn", "realmlp"]:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device
    return resolved_device


def check_device_estimator_combination(device: Literal["cpu", "cuda", "auto"], estimator: EstimatorType) -> None:
    '''Checks whether the specified estimator can be run on the specified device'''
    # uniform "es" and "base" estimator versions
    estimator = re.sub("^es_", "", estimator)

    if estimator in ["random_forest", "extra_trees"] and device == "cuda":
        raise DeviceError(f"cuda is not compatible with {estimator} estimator.")
    elif estimator in ["catboost", "xgb", "lgbm"] and device == "cuda":
        raise DeviceError(f"Metatab does not support {estimator} on GPU.")
    

def check_cuda_is_available() -> None:
    '''Checks whether cuda is available raising an error otherwise'''
    if not torch.cuda.is_available():
        raise DeviceError("cuda is requested but not available.")