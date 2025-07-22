from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

if TYPE_CHECKING:
    import logging



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



def get_repetition_fold(iteration: int, pars: dict) -> tuple: 
    '''
    Utility to get the repetition and fold info based on 
    the current iteration and splitting mode.
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



## TODO: remove configuration part when/if conf system is removed.
def log_program_setting(pars: dict, logger: logging.Logger, name_dataset: str) -> None:
    '''Logs info about the program input parameters/setting at debug level'''
    if pars["tune"]:
        conf_string = "c0" if pars["hps_configuration"] is None else pars["hps_configuration"]
        logger.debug(f"\nLaunching tuned {conf_string} {pars["estimator"]} on {name_dataset}!")
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


# def universal_predict_proba(
#     estimator: Classifier | Pipeline | GridSearchCV, 
#     X: np.ndarray | pd.DataFrame, 
#     **kwargs
# ) ->  np.ndarray:
#     '''
#     Adapter that invokes the estimators predict_proba method
#     allowing for additional parameters. 
#     Relies on the assumption that the estimator predict methods 
#     accept the X as first argument.
#     Necessary since AesFineTunedTabPFNClassifier does not 
#     respect the usual predicts signature.

#     Parameters:
#         estimator (Classifier | Pipeline | GridSearchCV):
#             The estimator object that calls the predict_proba method.
#         X (np.ndarray | pd.DataFrame):
#             The test data.
#         kwargs:
#             Other keyword arguments.
    
#     Returns:
#         np.ndarray: The predicted probabilities.
#     '''
#     if isinstance(estimator, AesFineTunedTabPFNClassifier):
#         return estimator.predict_proba(X, **kwargs)
#     else:
#         return estimator.predict_proba(X)