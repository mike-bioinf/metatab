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



def create_logger(stream) -> logging.Logger:
    '''
    Create a logger to a stream.
    Parameters:
        stream: Either sys.stdout or sys.stderr.
    Returns: The logger instance.
    '''
    logger = logging.getLogger("runtabpfn")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger



def fix_estimator_fixed_params_in_fit_program(
    estimator: Estimator,
    fit_program_params: dict
) -> None:
    '''
    Some estimators require their fixed parameters to be updated according fit program inputs. 
    This function performs that adjustment and stores the updated values in the instance-level 
    attribute fixed_params (while the "original" parameter remain defined at the class level).
    In case the estimator needs no adjustment then no attribute is created (nothing is done).

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
        out_file: Path = fit_program_params["output_path"].resolve()
        autogluon_models_folder = out_file.parent / f"autogluon_tabpfn_{out_file.stem}"
        new_fixed_params = _fix_autotabpfn_fixed_params_in_fit_program(
            new_fixed_params,
            autogluon_models_folder
        )
    else:
        # we avoid creating the instance-level attribute when not necessary
        return None

    estimator.fixed_params = new_fixed_params



def _fix_autotabpfn_fixed_params_in_fit_program(
    params: dict, 
    out_path: Path
) -> dict:
    '''
    Set the "path" parameter in the AutoTabPFNClassifier params dict.
    Returns the modified params dict.
    '''
    phe_init_args = params.get("phe_init_args", {})
    phe_init_args["path"] = out_path
    params["phe_init_args"] = phe_init_args
    return params



def fix_estimator_fixed_params_during_resampling(
    estimator: Estimator,
    repeat: int | NAType,
    fold: int,
    resample_program_params: dict
) -> None:
    '''
    Some estimators require their fixed parameters to be updated during resampling.
    This function performs that adjustment and stores the updated values in the instance-level 
    attribute fixed_params (while the "original" parameters remain defined at the class level).
    In case the estimator needs no adjustment then no attribute is created (nothing is done).

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
        out_path: Path = resample_program_params["output_path"].resolve()
        autogluon_top_folder = out_path / "autogluon_tabpfn"
        os.makedirs(autogluon_top_folder, exist_ok=True)
        new_fixed_params = _fix_autotabpfn_fixed_params_for_resampling_round(
            new_fixed_params,
            autogluon_top_folder,
            repeat, 
            fold
        )
    else:
        # we avoid creating the instance-level attribute when not necessary
        return None
    
    # setting the new params in an instance-level attribute
    estimator.fixed_params = new_fixed_params



def _fix_autotabpfn_fixed_params_for_resampling_round(
        params: dict, 
        autogluon_folder: Path, 
        repeat: int | NAType, 
        fold: int
    ):
    '''
    Adjust the "path" parameter of AutoTabPFNClassifier estimator
    according to the resample iteration. This assure a new folder
    in which autogluon saves the models for each resample iteration.
    '''
    # take the existing dict or create a new empty one
    phe_init_args = params.get("phe_init_args", {})

    # address holdout and cross-validation scenarios
    if repeat is pd.NA:
        folder_models = autogluon_folder / f"classifiers_iteration{fold}"
    else:
        folder_models = autogluon_folder /f"classifiers_repeat{repeat}_fold{fold}"

    phe_init_args["path"] = folder_models
    return phe_init_args