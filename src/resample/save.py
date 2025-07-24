import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

from resample.constants import (
    HPO_DICT_BASE_KEYS, 
    PRED_DATAFRAME_RESULTS_FIXED_COLUMNS
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



def create_dict_hpo(pars: dict) -> dict[str, list]:
    '''
    Creates the dictionary used to store the best hyperparameters info.
    The dict is empy when HP tuning is not requested.
    Note: Now it needs the hps dict in input.
    '''
    if not pars["tune"]: return {}
    return {
        key: [] 
        for key in HPO_DICT_BASE_KEYS + list(pars["tune_configuration"]["params_distributions"].keys())
    }



def populate_dict_lists_(dictionary: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the dictionary internal lists.'''
    for key, value in kwargs.items():
        dictionary[key].append(value)
        


def create_dict_results(pars: dict) -> dict:
    '''Utility to create the fillable dict result'''
    columns = (
        PRED_DATAFRAME_RESULTS_FIXED_COLUMNS + 
        get_preprocessing_columns_results(pars["preprocessing"])
    )
    return {key: [] for key in columns}



def get_preprocessing_columns_results(
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