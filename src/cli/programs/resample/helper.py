from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from pandas._libs.missing import NAType
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

if TYPE_CHECKING:
    from logging import Logger



def log_program_setting(pars: dict, logger: Logger, name_dataset: str) -> None:
    if pars["estimator_mode"] == "tune":
        logger.debug(
            f"\nLaunching {pars["tune_algo"]} tuned {pars["estimator"]} with {pars["tune_space"]} space on {name_dataset}!"
        )
    else:
       logger.debug(f"\nLaunching {pars["estimator"]} on {name_dataset}!")



def log_iteration(pars: dict, fold: int, repetition: int, logger: Logger) -> None:
    if pars["splitting_mode"] == "cv":
        logger.debug(f'Running on fold {fold} of repetition {repetition}:')
    elif pars["splitting_mode"] == "holdout":
        logger.debug(f'Running holdout iteration {fold}":')



def get_repetition_fold(iteration: int, pars: dict) -> tuple[int|NAType, int]: 
    '''
    Utility to get the repetition and fold info based 
    on the current iteration and splitting mode.
    Returns a binary tuple of int and/or pd.NA.
    '''
    if pars["splitting_mode"] == "cv":
        repetition = iteration // pars["n_cv_folds"]
        fold = iteration - (pars["n_cv_folds"] * repetition)
    elif pars["splitting_mode"] == "holdout":
        fold, repetition = iteration, pd.NA
    else:
        raise ValueError("Unsupported resampling mode.")
    
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
    seed = pars["seed_splitter"]
    
    if splitting_mode == "cv":
        splitter = RepeatedStratifiedKFold(
            n_splits=pars["n_cv_folds"], 
            n_repeats=pars["n_cv_repeats"], 
            random_state=seed
        )
    elif splitting_mode == "holdout":
        splitter = StratifiedShuffleSplit(
            n_splits=pars["n_holdout_splits"], 
            train_size=pars["holdout_train_size"], 
            random_state=seed
        )
    else:
        raise ValueError("Unsupported resampling mode.")
    
    return splitter



def populate_dict_lists_(dictionary: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the dictionary internal lists'''
    for key, value in kwargs.items():
        dictionary[key].append(value)



def create_json_configuration_file(pars: dict, filepath: str | Path) -> None:
    '''Create a json representation of the input program configuration'''
    corrected_pars = {}

    # Path object cannot be serialized in json
    for k, v in pars.items():
        corrected_pars[k] = str(v) if isinstance(v, Path) else v    
    
    with open(filepath, "w") as f:
        json.dump(corrected_pars, f, indent=4)



def silent_nanmin(a: np.ndarray) -> np.ndarray:
    '''numpy nanmin version which do not raise a warning when all array elements are nan'''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", 
            message=".*All-NaN slice encountered.*",
            category=RuntimeWarning
        )
        return np.nanmin(a)