from __future__ import annotations

import torch
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from metatab.preprocessing import create_classifier_pipeline
from metatab.utils.exceptions import DeviceError
from metatab.utils.general import ensure_or_create

if TYPE_CHECKING:
    from metatab.preprocessing.types import ResolvedPreprocessingStrategy
    from metatab.utils.types import Classifier, DefaultClassifierType
    from metatab.utils.types import XType, YType
    from sklearn.pipeline import Pipeline
    from metatab.classifiers.registry import ClassifierSpec



def create_pipeline(
    classifier_class: Classifier,
    classifier_params: dict,
    callbacks_on_classifier_params: None | list,
    y: YType,
    preprocessing: ResolvedPreprocessingStrategy,
    classifier_random_state_parameter: str | None,
    classifier_nthreads_paramater: str | None,
    classifier_device_parameter: str | None,
    seed: int,
    n_threads: int,
    device: Literal["cpu", "cuda"],
) -> Pipeline:
    '''
    Create the pipeline headed by the classifier. In detail:
    - finalize the classifier params with seed, n_threads, device info.
    - Calls the callbacks that adjust the params.
    - creates the classifier with the finalized params inside the pipeline.

    Notes: it create always a deepcopy of the dixed params on which to operate.

    Returns:
        Pipeline: The pipeline headed by the classifier.
    '''
    params = deepcopy(classifier_params)

    if classifier_random_state_parameter: 
        params[classifier_random_state_parameter] = seed

    if classifier_nthreads_paramater: 
        params[classifier_nthreads_paramater] = n_threads

    if classifier_device_parameter:
        params[classifier_device_parameter] = device
    
    for callback in ensure_or_create(callbacks_on_classifier_params, list):
        params = callback(params, y, False)

    pipe = create_classifier_pipeline(
        preprocessing=preprocessing,
        density_feature_selector_strategy="exact", ###REFACTOR: CAPIRE COME GESTIRLO (FISSO?)
        classifier=classifier_class,
        classifier_params=classifier_params,
    )

    return pipe



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
            f"'{type_classifier}' is early stopped. The 'validation_set' param cannot be None."
        )
    elif not clf_uses_validation_set and validation_set is not None:
        raise ValueError(
            f"'{type_classifier}' cannot be early stopped. The 'validation_set' param must be None."
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



def check_validation_set(validation_set: float | None) -> None:
    '''
    Check validation set type and value.
    '''
    if not isinstance(validation_set, float) and validation_set is not None:
        raise TypeError("'validation_set' must be a float or None.")

    if isinstance(validation_set, float):
        if not 0 < validation_set < 1:
            raise ValueError("'validation_set' must be a float in (0, 1).")



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