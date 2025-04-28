import argparse
import warnings
from pathlib import Path
from typing import Any
from ast import literal_eval
from finetabpfn import SKLEARN_WRAPPER_SECONDARY_PARAMS
from copy import deepcopy



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # positional arguments
    p.add_argument("input_path", help="Path to the dataset folder/file.")
    p.add_argument("output_path", help="Path of the folder storing the results. The folder is created.")

    # required options
    p.add_argument("-i", "--input-mode", required=True, choices=["sets", "xy", "df"],
                    help="Define the expected form of the input (must be one of 'sets', 'xy', or 'df')")

    p.add_argument("-s", "--splitting-mode", required=True, choices=["no", "holdout", "cv"],
                    help="Define the splitting strategy (must be one of 'no', 'holdout' or 'cv')")

    # other options
    p.add_argument("-d", "--splitting-specs", default=None, help=
                    "Specify splitting details as key=value pairs. \
                    Defaults to {'n_repeats':10, 'n_splits':5} if --splitting-mode is equal to cv.\
                    Defaults to {'n_splits':50, 'train_size':0.9} if --splitting-mode is equal to holdout.")

    p.add_argument("-p", "--preprocessing", default="no", choices=["no", "filter", "pca"], nargs="+",
                    help= "Preprocessing on the feature space. Choose one or more: 'no', 'filter' and 'pca'")

    p.add_argument("-m", "--model", default="base", choices=["base", "auto", "ft", "ft_opt", "rf"], 
                    help=""" ML 'model'. One of base, auto, ft, ft_opt and rf. 
                    Note that base and auto refers to TabPFN and AutoTabPFN, 
                    while ft and ft_opt to finetune and finetune with optimization.""")

    p.add_argument("-n", "--model-specs", default=None, 
                    help="""String represenatation of a dict of param value couples like "{'param': value, ...}" to pass to the model.
                    See the TabPFNClassifier, AutoTabPFNClassifier, FineTuneTabPFN, SklearnFineTuneTabPFN and RandomForestClassifier params for info.
                    In case of a "finetune model" ('ft' or 'ft_opt') one can/must pass the parameters of both FineTuneTabPFN and SklearnFineTuneTabPFN
                    classes. The correct ripartion is internally managed.""")

    p.add_argument("-t", "--test-dataset", default=None, 
                    help="Path to the folder/file (respects --input-mode) of a second dataset that is used as test data.")

    p.add_argument("-y", "--target-feature", default=None, 
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")

    p.add_argument("-g", "--grid-search", default=None, 
                    help="""dict formatted argument that will be passed to the 'param_grid' parameter of GridSearchCV class.
                    To use when model is 'rf' and when HPO is desired.""")

    p.add_argument("-r", "--seed", default=10, type=int, 
                    help="""Seed used to control randomness. 
                    In particular it controls the randomness inherent to the random forest model, splitting, ensemble and finetune procedures.
                    It does not control the randomness inherent to the base tabpfn model, which is controlled by a second random state that can be
                    set in "-n"/"--model-specs". This is true also for the base model finetuned in finetuning scenarios. Deafults to 10.""")

    return p.parse_args(args)



def check_args(pars: dict):
    '''Utility to check for incompatible args'''

    if pars["input_mode"] == "sets" and pars["splitting_mode"] != "no":
        raise ValueError("--splitting-mode must be 'no' when --input-mode is 'sets'")

    if pars["input_mode"] != "sets" and pars["test_dataset"] is None and pars["splitting_mode"] == "no":
        raise ValueError("with --input-mode 'xy' or 'df' and without --test-dataset, --splitting-mode cannot be 'no'")

    if pars["input_mode"] == "sets" and pars["test_dataset"] is not None:
        raise ValueError("with --input-mode 'sets' is not possible to use a second dataset as test")

    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("ERROR: -y must be specified when --input-mode equal to 'df'")
        
    if pars["test_dataset"] is not None and pars["splitting_mode"] != "no":
        raise ValueError("with a second dataset used as test --splitting-mode must be equal to 'no'")
    
    if pars["model"] in ["auto", "ft", "ft_opt"] and "pca" in pars["preprocessing"]:
        raise ValueError("Is not possible use the 'pca' preprocessing with 'auto', 'ft' and 'ft_opt' model.")



def adjust_args(pars: dict) -> dict:
    '''Utility to parse some arguments in python dict and to adjust some arguments value based on other arguments value.'''

    pars["input_path"] = Path(pars["input_path"])
    pars["output_path"] = Path(pars["output_path"])
    pars["test_dataset"] = Path(pars["test_dataset"]) if pars["test_dataset"] else pars["test_dataset"]

    if pars["splitting_mode"] == "no" and pars["splitting_specs"] is not None:
        warnings.warn("--splitting-specs is ignored since --splitting-mode is 'no'")
        pars["splitting_specs"] = None

    pars["model_specs"] = {} if pars["model_specs"] is None else try_parse_specs_into_dict(pars["model_specs"], "--model-specs")

    if pars["model"] in ["base", "auto"]:
        pars["model_specs"]["ignore_pretraining_limits"] = True

    if pars["grid_search"] is not None:
        pars["grid_search"] = try_parse_specs_into_dict(pars["grid_search"], "--grid-search")

    if pars["splitting_specs"] is not None:
        pars["splitting_specs"] = try_parse_specs_into_dict(pars["splitting_specs"], "--splitting-specs")
        specs = ["n_splits", "train_size"] if pars["splitting_mode"] == "holdout" else ["n_repeats", "n_splits"]
        for spec in specs:
            if spec not in pars["splitting_specs"].keys():
                raise ValueError(f"With '{pars["splitting_mode"]}' --splitting-mode you must pass the {specs} keys in --splitting-specs")

    if pars["splitting_mode"] == "cv" and pars["splitting_specs"] is None:
        pars["splitting_specs"] = {"n_repeats": 10, "n_splits": 5}

    if pars["splitting_mode"] == "holdout" and pars["splitting_specs"] is None:
        pars["splitting_specs"] = {"n_splits": 50, "train_size": 0.9}

    pars = adjust_args_for_finetune(pars)
    return pars



def adjust_args_for_finetune(pars: dict) -> dict:
    '''
    Adjust the parsed dict of arguments separating the params of the SklearnFineTuneTabPFN class
    to a separate dict. This dict is set to an empty dict if no finetuning is involved.
    '''
    model_specs = pars["model_specs"]
    ft_wrapper_specs = {}

    if pars["model"] in ["ft", "ft_opt"]:
        for key in SKLEARN_WRAPPER_SECONDARY_PARAMS:
            if key in model_specs:
                ft_wrapper_specs[key] = deepcopy(model_specs[key])
                del model_specs[key]
    
    pars["ft_wrapper_specs"] = ft_wrapper_specs
    return pars



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
