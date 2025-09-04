from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING
from estimators import MyAutoTabPFNClassifier, Estimator

if TYPE_CHECKING:
    from pandas._libs.missing import NAType



def check_y_is_integer_encoded(y: pd.Series, is_predict_scenario: bool = False) -> None:
    '''
    Checks that y is integer encoded. 
    This is essential to avoid errors in metrics computation.
    Raises different error messages depending on the scenario.
    '''
    y = np.asarray(y)
    
    if not np.issubdtype(y.dtype, np.integer):
        message = "Target variable y must be integer-encoded (e.g., 0, 1, 2, ...)."
        if is_predict_scenario:
            message += (
                " Note: in binary classification, class `1` is treated as the reference class"
                " in performance metrics computation."
            )
        raise ValueError(message)



class FlushStreamHandler(logging.StreamHandler):
    '''
    A stream handler that flush when emits.
    Useful to deliver real time logging in HPC environment.
    '''
    def emit(self, record):
        super().emit(record)
        super().flush()


def create_logger(stream) -> logging.Logger:
    '''
    Create a logger to a stream.
    Parameters:
        stream: Either sys.stdout or sys.stderr.
    Returns: The logger instance.
    '''
    logger = logging.getLogger("metatab")
    logger.setLevel(logging.DEBUG)
    stream_handler = FlushStreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger



def fix_estimator_fixed_params_in_fit_program_(
    estimator: Estimator,
    fit_program_params: dict
) -> None:
    '''
    Some estimators require their fixed parameters to be updated according fit program inputs. 
    This function performs that adjustment and stores the updated values in the instance-level 
    attribute fixed_params (while the "original" parameter remain defined at the class level).
    In case the estimator needs no adjustment a deepcopy of the class-level attribute is 
    created at instance level.

    Parameters:
        estimator (Estimator): 
            The estimator instance.
        fit_program_params (dict):
            Dictionary containing the input parameters for the fit program.
    '''
    new_fixed_params = deepcopy(estimator.fixed_params)

    if isinstance(estimator, MyAutoTabPFNClassifier):
        # we create the autogluon directory in parent folder of the output file
        # using the root of the output filename in its name to prevent overwriting issues 
        out_file: Path = fit_program_params["output_path"]
        autogluon_models_folder = out_file.parent / f"autogluon_tabpfn_{out_file.stem}"
        new_fixed_params = _add_autogluon_path_to_params(
            params=new_fixed_params,
            path=autogluon_models_folder,
            repeat=None,
            fold=None
        )

    # setting the new params in an instance-level attribute
    estimator.fixed_params = new_fixed_params



def fix_estimator_fixed_params_during_resampling_(
    estimator: Estimator,
    repeat: int | NAType,
    fold: int,
    resample_program_params: dict
) -> None:
    '''
    Some estimators require their fixed parameters to be updated during resampling.
    This function performs that adjustment and stores the updated values in the instance-level 
    attribute fixed_params (while the "original" parameters remain defined at the class level).
    In case the estimator needs no adjustment a deepcopy of the class-level attribute is 
    created at instance level.

    Parameters:
        estimator (Estimator): 
            The estimator instance.
        
        repeat (int | NAType): 
            The resampling repeat. 
            An integer if the strategy is cross-validation, or Na if it is holdout.
        
        fold (int): 
            The resampling fold. 
            Represents the inner iteration within a repeat for cross-validation, 
            or the general iteration in holdout.

        resample_program_params (dict): 
            Dictionary containing the input parameters for the resample program.  
    '''
    new_fixed_params = deepcopy(estimator.fixed_params)

    if isinstance(estimator, MyAutoTabPFNClassifier):
        out_path: Path = resample_program_params["output_path"]
        autogluon_top_folder = out_path / "autogluon_tabpfn"
        os.makedirs(autogluon_top_folder, exist_ok=True)
        new_fixed_params = _add_autogluon_path_to_params(
            new_fixed_params,
            autogluon_top_folder,
            repeat, 
            fold
        )

    # setting the new params in an instance-level attribute
    estimator.fixed_params = new_fixed_params



def _add_autogluon_path_to_params(
    params: dict, 
    path: Path,
    repeat: int | NAType | None,
    fold: int | None
) -> dict:
    '''
    Add the "path" parameter in the AutoTabPFNClassifier params dict,
    needed by autogluon to save the fitted classifiers.
    Distinguishes fit and resample cv/houldout scenarios 
    based on the type of repeat and fold argument.
    Returns the modified params dict.
    '''
    phe_init_args = params.get("phe_init_args", {})

    if repeat is None and fold is None:
        # we are in the fit program
        folder_models = path
    elif repeat is pd.NA and isinstance(fold, int):
        # we are in the resample program with holdout
        folder_models = path / f"classifiers_iteration{fold}"
    elif isinstance(repeat, int) and isinstance(fold, int):
        # we are in the resample program with cv
        folder_models = path /f"classifiers_repeat{repeat}_fold{fold}"
    else:
        raise ValueError(
            "Unrecognazible combination of types for repeat and fold arguments."
        )
    
    phe_init_args["path"] = folder_models
    params["phe_init_args"] = phe_init_args
    return params
