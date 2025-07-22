import os
from pathlib import Path



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



def check_fit_args(pars: dict) -> None:
    '''
    General check on fitting arguments. 
    Check used both for fit and resample programs
    '''
    check_target_feature(pars)
    check_not_tunable_estimators(pars)



def check_target_feature(pars: dict) -> None:
    '''Check that the target feature is set with df input-mode'''
    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("--target-feature must be specified with 'df' input mode.")



## TODO: complete with the tune-tabpfn estimator name
def check_not_tunable_estimators(pars: dict) -> None:
    '''Check whether the tune flag is used with not tunable estimator'''
    if pars["tune"] and pars["estimator"] == "tabpfn":
        raise ValueError(
            "The 'tabpfn' estimator cannot be tuned setting --tune. Use the '' estimator."
        )


## TODO: maybe to remove in production when a fixed conf is used for each estimator.
def check_ambiguous_tune_setting(pars: dict) -> None:
    '''
    Check whether a configuration of HPs is passed 
    to tunable estimators with the tune flag down.
    '''
    if not pars["tune"] and pars["hps_configuration"] is not None:
        raise ValueError(
            "A tuning configurations is passed but tuning is not requested."
        )


def check_incompatible_estimator_preprocessing(pars: dict) -> None:
    pass

