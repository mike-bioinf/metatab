import os
from pathlib import Path
from typing import Any
from ast import literal_eval

from estimators.estimators.params import (
    DEFAULT_TUNE_CONFIGURATION, 
    RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS,
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_1
)



def adjust_io_paths_(pars: dict, input_arg: str, output_arg: str) -> None:
    '''
    Convert paths to Path objects.
    The function works in place.
    '''
    pars[input_arg] = Path(pars[input_arg])
    pars[output_arg] = Path(pars[output_arg])



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
    '''General check on fit and resample program arguments'''
    check_target_feature(pars)
    check_not_tunable_estimators(pars)
    check_ambiguous_tune_setting(pars)



def check_target_feature(pars: dict) -> None:
    '''Check that the target feature is set with df input-mode'''
    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("--target-feature must be specified with 'df' input mode.")



## TODO: complete with the tune-tabpfn estimator name
def check_not_tunable_estimators(pars: dict) -> None:
    '''Check whether the tune flag is used with not tunable estimators'''
    if pars["tune"] and pars["estimator"] == "tabpfn":
        raise ValueError(
            "The 'tabpfn' estimator cannot be tuned setting --tune. Use the '' estimator."
        )


## TODO: maybe to remove in production when a fixed conf is used for each estimator.
def check_ambiguous_tune_setting(pars: dict) -> None:
    '''Check whether a tune configuration is passed with the tune flag down.'''
    if not pars["tune"] and pars["tune_configuration"] is not None:
        raise ValueError(
            "A tuning configurations is passed but tuning is not requested."
        )



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
    
    # fill the missing keys with defaults
    if do_actions:
        for key, value in DEFAULT_TUNE_CONFIGURATION.items():
            if not key in pars["tune_configuration"].keys():
                pars["tune_configuration"][key] = value

    # check for unsupported keys
    if do_actions:
        if len(pars["tune_configuration"]) > len(DEFAULT_TUNE_CONFIGURATION):
            raise ValueError(
                "Passed unsupported keys in 'tune_configuration' argument."
            )

    # add the parameters distributions
    pars["tune_configuration"]["params_distributions"] = _pick_params_distributions_configuration(pars)



def _pick_params_distributions_configuration(pars: dict) -> dict | None:
    conf = pars["tune_configuration"]["configuration"]
    estimator = pars["estimator"]

    match (estimator, conf):
        case ("random_forest", "c0"):
            return RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS
        case ("xgb" | "es_xgb", "c0"):
            return RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS
        case ("xgb" | "es_xgb", "c1"):
            return RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_1 
        case _:
            raise ValueError(
                f"Unsupported configuration '{conf}' for '{estimator}' estimator."
            )



def try_parse_specs_into_dict(specs: str, error_message_specs: str) -> dict[str, Any]:
    '''Utility to parse the string dict representation to a dict'''
    try:
        specs = literal_eval(specs)
    except Exception:
        raise ValueError(
            f"{error_message_specs} " + "cannot be correctly parsed into a dict. \
            It should be passed following the syntax '{'key': value, ...}'.\
            Remember to enclose the keys in ticks ('') if they are python strings."
        )
    return specs



def check_incompatible_estimator_preprocessing(pars: dict) -> None:
    pass
