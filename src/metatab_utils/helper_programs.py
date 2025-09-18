from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING
from ast import literal_eval
from estimators import Estimator

from estimators.constants import (
    EARLY_STOPPED_ESTIMATORS, 
    NON_TUNABLE_ESTIMATORS,
    PCA_INCOMPATIBLE_ESTIMATORS
)

from estimators.params import (
    DEFAULT_TUNE_CONFIGURATION,
    DEFAULT_ESTIMATORS_TUNE_SPACES,
    TuningParams
)

from estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyAutoTabPFNClassifier,
    MyAesFineTunedTabPFNClassifier
)

if TYPE_CHECKING:
    from pandas._libs.missing import NAType




def adjust_io_paths_(pars: dict, input_arg: str, output_arg: str) -> None:
    '''
    Convert paths to absolute Path objects.
    The function works in place.
    '''
    pars[input_arg] = Path(pars[input_arg]).resolve()
    pars[output_arg] = Path(pars[output_arg]).resolve()



def manage_output_path(pars: dict, output_arg: str, is_folder: bool) -> None:
    '''
    Control whether the output folder exists and whether to create it if not.
    One must specify the output parameter and if this is expected
    to be a folder. If not the parent folder is considered.
    Assumes that the output argument is a Path object.
    '''
    out: Path = pars[output_arg]
    out_folder = out if is_folder else out.parent

    if not out_folder.exists() and not pars["create_outdir"]:
        raise FileNotFoundError(f"{out_folder} does not exists!")
    elif not out_folder.exists() and pars["create_outdir"]:
        os.makedirs(out_folder)



def check_fit_resample_args(pars: dict) -> None:
    '''General check for fit and resample program arguments'''
    check_target_feature(pars)
    check_not_tunable_estimators(pars)
    check_ambiguous_tune_setting(pars)
    check_incompatible_estimator_preprocessing(pars)


def check_target_feature(pars: dict) -> None:
    '''Check that the target feature is set with df input-mode'''
    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("--target-feature must be specified with 'df' input mode.")


def check_not_tunable_estimators(pars: dict) -> None:
    '''Check whether the tune flag is used with not tunable estimators'''
    if (estimator := pars["estimator"]) in NON_TUNABLE_ESTIMATORS and pars["tune"]:
        raise ValueError(f"Estimator '{estimator} cannot be tuned.'")
        

def check_ambiguous_tune_setting(pars: dict) -> None:
    '''Check whether a tune configuration is passed with the tune flag down.'''
    if not pars["tune"] and pars["tune_configuration"] is not None:
        raise ValueError(
            "A tuning configurations is passed (tune_configuration is not None)" +
            " but tuning is not requested (tune flag down)."
        )


def check_incompatible_estimator_preprocessing(pars: dict) -> None:
    if (
        (estimator := pars["estimator"]) in PCA_INCOMPATIBLE_ESTIMATORS and 
        pars["preprocessing"] == "pca"
    ):
        raise ValueError(f"PCA preprocessing cannot be used with '{estimator}' estimator.")


def check_tune_algo(pars: dict) -> None:
    '''Check on the validity of the tuning algo'''
    if not pars["tune"] or pars["tune_configuration"] is None:
        return None
    else:
        input_tune_algo = pars["tune_configuration"]["algo"]
        if input_tune_algo not in ["random", "tpe", "meta_tpe", "meta"]:
            raise ValueError(
                "The tuning search algorithm must be one of 'random', 'tpe', 'meta_tpe' or 'meta'." +
                f" Currently {input_tune_algo}."
            )


def check_y_is_integer_encoded(y: pd.Series, is_predict_scenario: bool = False) -> None:
    '''
    Checks that y is integer encoded. 
    This is essential to avoid errors in metrics computation.
    Raises different error messages depending on the scenario.
    '''
    y = y.to_numpy()

    if not np.issubdtype(y.dtype, np.integer):
        message = "Target variable y must be integer-encoded (e.g., 0, 1, 2, ...)."
        if is_predict_scenario:
            message += (
                " Note: in binary classification, class `1` is treated as the reference class"
                " in performance metrics computation."
            )
        raise ValueError(message)
    


def adjust_early_stopping_rounds_(pars: dict) -> None:
    estimator = pars["estimator"]
    early_stopping_rounds = pars["early_stopping_rounds"]

    if early_stopping_rounds != -1 and estimator not in EARLY_STOPPED_ESTIMATORS:
        return ValueError(
            "'early_stopping_rounds' must be -1 when using a non early stopped estimator."
        )
    
    # set the default
    if early_stopping_rounds < 0 and estimator in EARLY_STOPPED_ESTIMATORS:
        pars["early_stopping_rounds"] = 100
    


def adjust_tune_configuration_arg_(pars: dict) -> None:
    '''
    Adjust the tune configuration argument.
    The argument is left as is if tuning is not requested.
    Modifies the dict of arguments in place.
    '''
    if not pars["tune"]:
        return None
    
    do_actions = True

    # set the default or parse the string into a dict
    if pars["tune_configuration"] is None:
        pars["tune_configuration"] = DEFAULT_TUNE_CONFIGURATION
        do_actions = False
    else:
        pars["tune_configuration"] = try_parse_specs_into_dict(
            pars["tune_configuration"], 
            "tune_configuration"
        )
    
    if do_actions:
        # fill the missing keys with defaults
        actual_keys = pars["tune_configuration"].keys()
        for key, value in DEFAULT_TUNE_CONFIGURATION.items():
            if not key in actual_keys:
                pars["tune_configuration"][key] = value

        # check for unsupported keys
        if len(pars["tune_configuration"]) > len(DEFAULT_TUNE_CONFIGURATION):
            raise ValueError(
                "Passed unsupported keys in 'tune_configuration' argument."
            )

    # add the parameters distributions
    pars["tune_configuration"]["params_distributions"] = (
        _pick_params_distributions_configuration(pars)
    )



def try_parse_specs_into_dict(specs: str, error_message_specs: str) -> dict[str, Any]:
    '''Utility to parse the string dict representation to a dict'''
    try:
        specs = literal_eval(specs)
    except Exception:
        raise ValueError(
            f"{error_message_specs} " + "cannot be correctly parsed into a dict." +
            "It should be passed following the syntax {'key': value, ...}, enclosing the dict with double quotes" +
            "Remember to enclose the keys in ticks ('') if they are python strings."
        )
    return specs



def _pick_params_distributions_configuration(pars: dict) -> dict:
    conf = pars["tune_configuration"]["configuration"]
    estimator = pars["estimator"]

    match (estimator, conf):
        case ("random_forest", "c0"):
            return TuningParams.RF_C0
        case ("random_forest", "c1"):
            return TuningParams.RF_C1
        
        case ("xgb" | "es_xgb", "c0"):
            return TuningParams.XGB_C0
        case ("xgb" | "es_xgb", "c1"):
            return TuningParams.XGB_C1
        case ("xgb" | "es_xgb", "c2"):
            return TuningParams.XGB_C2
        case ("xgb" | "es_xgb", "c3"):
            return TuningParams.XGB_C3
        case ("xgb" | "es_xgb", "c4"):
            return TuningParams.XGB_C4
        
        case ("catboost" | "es_catboost", "c0"):
            return TuningParams.CATBOOST_C0
        case ("catboost" | "es_catboost", "c1"):
            return TuningParams.CATBOOST_C1
        case ("catboost" | "es_catboost", "c2"):
            return TuningParams.CATBOOST_C2
        case ("catboost" | "es_catboost", "c3"):
            return TuningParams.CATBOOST_C3
        case ("catboost" | "es_catboost", "c4"):
            return TuningParams.CATBOOST_C4
        case ("catboost" | "es_catboost", "c5"):
            return TuningParams.CATBOOST_C5

        case ("lgbm" | "es_lgbm", "c0"):
            return TuningParams.LGMB_C0
        case ("lgbm" | "es_lgbm", "c1"):
            return TuningParams.LGMB_C1

        case ("tabpfn", "c0"):
            return TuningParams.TABPFN_C0
        
        case (_, "default"):
            return DEFAULT_ESTIMATORS_TUNE_SPACES[estimator]
            
        case _:
            raise ValueError(
                f"Unsupported configuration '{conf}' for '{estimator}' estimator."
            )



def pick_estimator_class(pars: dict) -> Estimator:
    match (pars["estimator"], pars["tune"]):
        case ("random_forest", False):
            return MyRandomForestClassifier
        case ("random_forest", True):
            return MyTunedRandomForestClassifier
        
        case ("xgb", False):
            return MyXGBClassifier
        case ("xgb", True):
            return MyTunedXGBClassifier
        case ("es_xgb", False):
            return MyESXGBClassifier
        case ("es_xgb", True):
            return MyTunedESXGBClassifier
        
        case("catboost", False):
            return MyCatBoostClassifier
        case("catboost", True):
            return MyTunedCatBoostClassifier
        case ("es_catboost", False):
            return MyESCatBoostClassifier
        case("es_catboost", True):
            return MyTunedESCatBoostClassifier
        
        case ("lgbm", False):
            return MyLGBMClassifier
        case("lgbm", True):
            return MyTunedLGBMClassifier
        case ("es_lgbm", False):
            return MyESLGBMClassifier
        case ("es_lgbm", True):
            return MyTunedESLGBMClassifier

        case ("tabpfn", False):
            return MyTabPFNClassifier
        case("tabpfn", True):
            return MyTunedTabPFNClassifier
        case("autotabpfn", _):
            return MyAutoTabPFNClassifier
        case("finetunetabpfn", _):
            return MyAesFineTunedTabPFNClassifier
    
        case _:
            raise ValueError("Unsupported estimator.")
        


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