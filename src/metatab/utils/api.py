from __future__ import annotations

import torch
import pandas as pd
from typing import TYPE_CHECKING, Literal
from sklearn.preprocessing import LabelEncoder
from metatab.utils.exceptions import DeviceError

if TYPE_CHECKING:
    from metatab.utils.types import DefaultClassifierType
    from metatab.utils.types import XType, YType
    from metatab.classifiers.registry import ClassifierSpec



def encode_y(X: XType, y: YType) -> tuple[LabelEncoder, YType]:
    '''
    Encode the y using sklearn LabelEncoder.
    The X is needed only to uniform the resulting y type to X.
    Returns the sklearn label encoder and the encoded y. 
    '''
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    if isinstance(X, pd.DataFrame):
        name = y.name if isinstance(y, pd.Series) else None
        return le, pd.Series(y_encoded, name=name)
    return le, y_encoded


## REFACTOR: to cancel
def check_validation_set_classifier_combination(
    validation_set: None | float | tuple[XType, YType],
    classifier_spec: ClassifierSpec,
    type_classifier: DefaultClassifierType
) -> None:
    '''
    Check whether the classifier needs or refuses the validation set info.
    '''
    clf_uses_validation_set = classifier_spec.early_stop_on_validation_set

    if clf_uses_validation_set and validation_set is None:
        raise ValueError(
            f"'{type_classifier}'uses validation set. The 'validation_set' param cannot be None."
        )
    elif not clf_uses_validation_set and validation_set is not None:
        raise ValueError(
            f"'{type_classifier}' does not use validation set. The 'validation_set' param must be None."
        )



# def check_validation_set(validation_set: float | tuple[XType, YType] | None) -> None:
#     '''
#     Check validation set type and value.
#     '''
#     if isinstance(validation_set, float):
#         if not 0 < validation_set < 1:
#             raise ValueError("'validation_set' must be a float in (0, 1).")

#     elif isinstance(validation_set, tuple):
#         if len(validation_set) != 2:
#             raise ValueError("validation_set must be a tuple (X, y)")
        
#         X, y = validation_set
        
#         if not isinstance(X, (pd.DataFrame, np.ndarray)):
#             raise TypeError(
#                 "The first element (X) of 'validation_set' must be a pd.DataFrame or np.ndarray"
#             )
        
#         if not isinstance(y, (pd.Series, np.ndarray)):
#             raise TypeError(
#                 "The second element (y) of 'validation_set' must be a pd.Series or np.ndarray"
#             )

#         if X.shape[0] != len(y):
#             raise ValueError("validation_set Xy shape mismatch: X rows != y length")

#     elif validation_set is None:
#         pass

#     else:
#         raise TypeError("'validation_set' must be None, a float, or a tuple (X, y)")



def check_validation_set(validation_set_size: float) -> None:
    '''Check validation set type and value'''
    if not isinstance(validation_set_size, float):
        raise TypeError("'validation_set_size' must be a float.")
    if not 0 < validation_set_size < 1:
        raise ValueError("'validation_set_size' must be a float in (0, 1).")



def check_device(
    device: Literal["auto", "cpu", "cuda"], 
    classifier_specs: list[ClassifierSpec]
) -> None:
    '''
    Check that the device info is compatible for all classifiers.
    In detail performs checks when device is "cuda".
    When "cpu" or "auto" no check is done since all classifiers support both.
    '''
    if device not in ["auto", "cpu", "cuda"]:
        raise DeviceError("Device must me one of: 'cpu', 'cuda', or 'auto'.")
    
    if device == "cuda":
        if not torch.cuda.is_available():
            DeviceError("cuda is requested but not available.")
        for classifier_spec in classifier_specs:
            if device not in classifier_spec.supported_devices:
                raise DeviceError(f"'{classifier_spec.type_classifier}' does not support cuda.")


## REFACTOR: to cancel
def handle_device(
    input_device: str, 
    classifier_spec: ClassifierSpec,
    type_classifier: DefaultClassifierType
) -> Literal["cpu", "cuda"]:
    '''
    Check and resolve the device info.
    Returns the resolved device ("cpu" or "cuda").
    '''
    clf_supported_devices = classifier_spec.supported_devices + ["auto"]

    if input_device not in clf_supported_devices:
        raise DeviceError(f"'{type_classifier}' supports: '{clf_supported_devices}'. Actually '{input_device}'.")
     
    # resolve "auto"
    if input_device == "auto":
        if classifier_spec.main_device == "cuda" and torch.cuda.is_available():
            input_device = "cuda"
        elif "cpu" in classifier_spec.supported_devices:
            input_device = "cpu"
        else:
            input_device = classifier_spec.supported_devices[0] # fallback to first supported

    # validate resolved device
    if input_device == "cuda" and not torch.cuda.is_available():
        raise DeviceError("cuda is requested but not available.")

    return input_device