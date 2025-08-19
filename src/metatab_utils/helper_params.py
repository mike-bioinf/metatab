import os
from pathlib import Path
from typing import Any
from ast import literal_eval
from estimators import Estimator

from estimators.params import (
    DEFAULT_TUNE_CONFIGURATION,
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
    MyTabPFNClassifier
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
    '''General check for fit and resample program arguments'''
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


def check_ambiguous_tune_setting(pars: dict) -> None:
    '''Check whether a tune configuration is passed with the tune flag down.'''
    if not pars["tune"] and pars["tune_configuration"] is not None:
        raise ValueError(
            "A tuning configurations is passed (tune_configuration is not None)" +
            " but tuning is not requested (tune flag down)."
        )


def check_tune_algo(pars: dict) -> None:
    '''Check on the validity of the tuning algo'''
    if not pars["tune"] or pars["tune_configuration"] is None:
        return None
    else:
        input_tune_algo = pars["tune_configuration"]["algo"]
        if input_tune_algo not in ["random", "tpe"]:
            raise ValueError(
                f"The tuning search algorithm must be one of 'random' or 'tpe'. Currently {input_tune_algo}."
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
            f"{error_message_specs} " + "cannot be correctly parsed into a dict. \
            It should be passed following the syntax '{'key': value, ...}'.\
            Remember to enclose the keys in ticks ('') if they are python strings."
        )
    return specs



def _pick_params_distributions_configuration(pars: dict) -> dict | None:
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

        case _:
            raise ValueError(
                f"Unsupported configuration '{conf}' for '{estimator}' estimator."
            )



def check_incompatible_estimator_preprocessing(pars: dict) -> None:
    pass



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
        
        case ("tabpfn", _):
            return MyTabPFNClassifier
        
        case _:
            raise ValueError("Unsupported estimator.")