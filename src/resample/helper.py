from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from pandas._libs.missing import NAType
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

if TYPE_CHECKING:
    import logging




def log_program_setting(pars: dict, logger: logging.Logger, name_dataset: str) -> None:
    '''Logs info about the program input parameters/setting at debug level'''
    if pars["tune"]:
        logger.debug(
            f"\nLaunching tuned {pars["tune_configuration"]["configuration"]} {pars["estimator"]} on {name_dataset}!"
        )
    else:
       logger.debug(f"\nLaunching {pars["estimator"]} on {name_dataset}!")



def log_iteration(pars: dict, fold: int, repetition: int, logger: logging.Logger) -> None:
    '''Utility that logs info about the current iteration at debug level'''
    if pars["splitting_mode"] == "cv":
        logger.debug(f'Running on fold {fold} of repetition {repetition}:')
    elif pars["splitting_mode"] == "holdout":
        logger.debug(
            f'Running holdout iteration {fold}, with train size {pars["splitting_specs"]["train_size"]}:'
        )



def get_repetition_fold(iteration: int, pars: dict) -> tuple[int|NAType, int]: 
    '''
    Utility to get the repetition and fold info based 
    on the current iteration and splitting mode.
    Returns a binary tuple of int and/or pd.NA.
    '''
    if pars["splitting_mode"] == "cv":
        repetition = iteration // pars["splitting_specs"]["n_splits"]
        fold = iteration - (pars["splitting_specs"]["n_splits"] * repetition)
    elif pars["splitting_mode"] == "holdout":
        fold, repetition = iteration, pd.NA
    else:
        raise ValueError("Unsupported splitting_mode.")
    
    return repetition, fold



def get_resample_iteration_signature(repeat: int | NAType, fold: int) -> str:
    '''
    Get the iteration signature as string, that is: 
    - {repeat}{fold} in cross-validation
    - {fold} in holdout
    '''
    if pd.isna(repeat):
        return f"{repeat}{fold}"
    else:
        return f"{fold}"



def get_iteration_estimator_filepath(pars: dict, repeat: int | NAType, fold: int) -> str:
    '''Get the filename made of the estimator name and current iteration info'''
    resample_iteration_signature = get_resample_iteration_signature(repeat, fold)
    filename = f"{pars["estimator"]}_{resample_iteration_signature}.pkl"
    return f"{pars["output_dir"]}/estimators/{filename}"



def pick_splitter(pars: dict):
    '''Utility to pick and create the right splitter depending on splitting_mode'''
    splitting_mode = pars["splitting_mode"]
    splitting_specs = pars["splitting_specs"]
    seed = pars["seed"]
    
    if splitting_mode == "cv":
        splitter = RepeatedStratifiedKFold(
            n_splits=splitting_specs["n_splits"], 
            n_repeats=splitting_specs["n_repeats"], 
            random_state=seed
        )
    elif splitting_mode == "holdout":
        splitter = StratifiedShuffleSplit(
            n_splits=splitting_specs["n_splits"], 
            train_size=splitting_specs["train_size"], 
            random_state=seed
        )
    else:
        raise ValueError("Unsupported splitting_mode.")
    
    return splitter



def populate_dict_lists_(dictionary: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the dictionary internal lists'''
    for key, value in kwargs.items():
        dictionary[key].append(value)



def create_json_configuration_file(pars: dict, filepath: str | Path) -> None:
    '''
    Create a json file riassuming the input program configuration.
    Takes in input the parsed and adjusted dict of program parameters.
    '''
    copy_pars = deepcopy(pars)

    if copy_pars["tune"]:
        del copy_pars["tune_configuration"]["params_distributions"]

    for key, value in copy_pars.items():
        if isinstance(value, Path):
            copy_pars[key] = str(value)
    
    with open(filepath, "w") as f:
        json.dump(copy_pars, f, indent=4)



def silent_nanmin(a: np.ndarray) -> np.ndarray:
    '''numpy nanmin version which do not raise a warning when all array elements are nan'''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", 
            message=".*All-NaN slice encountered.*",
            category=RuntimeWarning
        )
        return np.nanmin(a)