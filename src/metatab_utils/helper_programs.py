import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
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





def adjust_io_paths_(pars: dict, input_arg: str, output_arg: str) -> None:
    '''
    Convert paths to absolute Path objects.
    The function works in place.
    '''
    pars[input_arg] = Path(pars[input_arg]).resolve()
    pars[output_arg] = Path(pars[output_arg]).resolve()



def adjust_paths_(pars: dict, *args) -> None:
    '''
    Convert the values associated to the key specified in `args`
    to absolute Path object. The function works in place.
    Is a version of `adjust_io_paths_` that works on multiple args
    '''
    for arg in args:
        pars[arg] = Path(pars[arg]).resolve()



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
        raise ValueError(f"Estimator '{estimator}' cannot be tuned.")
        


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



def check_tune_configuration(pars: dict, logger: logging.Logger) -> None:
    '''
    General check on the tune configuration.
    In detail checks that:
    1) A valid algo is selected.
    2) Meta algo is selected with the default tune space.
    '''
    # do nothing when tune is not requested or used with our default configuration
    if not pars["tune"] or pars["tune_configuration"] is None:
        return None
    
    estimator = pars["estimator"]
    estimator_default_space = DEFAULT_ESTIMATORS_TUNE_SPACES[estimator][0]
    preprocessing = pars["preprocessing"]
    opt_algo = pars["tune_configuration"]["algo"]
    space = pars["tune_configuration"]["configuration"]

    if opt_algo not in ["random", "tpe", "meta"]:
        raise ValueError(
            "The tuning search algorithm must be one of 'random', 'tpe' or 'meta'." +
            f" Currently {opt_algo}."
        )

    if opt_algo == "meta" and space not in ["default", estimator_default_space]:
        raise ValueError(
            "'meta' algo can be used only with the estimator default tune space" + 
            f" ({estimator} --> {estimator_default_space})."
        )

    if (
        opt_algo == "meta" and
            (
                (estimator == "tabpfn" and preprocessing != "density_filter") or
                (estimator != "tabpfn" and preprocessing != "base")
            )
        ):
        logger.debug(
            "'meta' tuning algo is less effective when the following estimator-preprocessing couples are NOT respected:" +
            " tabpfn --> density_filter," +
            " others estimators --> base."
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
            return DEFAULT_ESTIMATORS_TUNE_SPACES[estimator][1]
            
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