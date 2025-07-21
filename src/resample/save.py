import json
from copy import deepcopy
from pathlib import Path
from typing import Literal
from resample.constants import HPO_DICT_BASE_KEYS

from estimators.estimators.params import (
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
    RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS
)



def get_estimator_filepath(pars: dict, repetition: int, fold: int) -> str:
    '''Get the filename to save the estimator'''
    splitting_mode = pars["splitting_mode"]

    if splitting_mode == "cv":
        n = f"{repetition}{fold}"
    elif splitting_mode == "holdout":
        n = f"{fold}"
    else:
        raise ValueError("Unsupported splitting_mode.")
    
    return f"{str(pars["output_dir"])}/estimators/{pars["estimator"]}__{pars["preprocessing"]}__{n}.pkl"



def create_dict_hpo(pars: dict, base_keys: list[str] = HPO_DICT_BASE_KEYS) -> dict[str, list]:
    '''
    Creates the dictionary used to store the best hyperparameters info.
    The dict is created only when HP are tuned. 
    In negative cases returns an empty dict.
    '''
    if not pars["tune"]:
        return {}
    hpo_specific_keys = get_hpo_names(pars)
    return {key: [] for key in base_keys + hpo_specific_keys}



def get_hpo_names(pars: dict) -> list[str]:
    '''
    Get the tunable HP names from the program input.
    If no tuning is involved returns an empy list.
    '''
    match (pars["estimator"], pars["tune"]):
        case ("random_forest", False):
            return []
        case ("random_forest", True):
            return list(RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS.keys())
        case ("xgb", False):
            return []
        case ("xgb", True):
            return list(RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS.keys())
        case ("ex_xgb", False):
            return []
        case ("ex_xgb", True):
            return list(RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS.keys())
        case ("tabpfn", _):
            return []
        case _:
            raise ValueError("Unsupported estimator.")



def populate_dict_hpo_(dict_hpo: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the "dict_hpo" internal lists.'''
    for key, value in kwargs:
        dict_hpo[key].append(value)
        


def create_dict_results(columns: list[str]) -> dict:
    '''Utility to create the fillable dict result'''
    return {key: [] for key in columns}



def get_additional_columns_results(
        preprocessing: Literal["base", "density_filter", "pca"]
    ) -> list[str]:
    '''
    Get the additional columns names for the prediction
    dataframe created from the "dict_results" according 
    to the preprocessing scenario.
    Returns a list of strings.
    '''
    if preprocessing == "base":
        add_cols = []
    elif preprocessing == "density_filter":
        add_cols = [
            "density_selection_strategy",
            "n_target_features",
            "minimum_selected_density_score"
        ]
    elif preprocessing == "pca":
        add_cols = [
            "n_pca_components",
            "explained_variance_ratio",
            "total_explained_variance_ratio"
        ]
    else:
        raise ValueError("Unsupported preprocessing.")

    return add_cols



def populate_dict_result_(dict_results: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the "dict_results" internal lists.'''
    for key, value in kwargs.items():
        dict_results[key].append(value)



def create_json_configuration_file(pars: dict, filepath: str | Path) -> None:
    '''
    Create a json file riassuming the input program configuration.
    Takes in input the parsed and adjusted dict of program parameters.
    '''
    copy_pars = deepcopy(pars)

    for key, value in copy_pars.items():
        if isinstance(value, Path):
            copy_pars[key] = str(value)
    
    with open(filepath, "w") as f:
        json.dump(copy_pars, f, indent=4)