import json
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Literal
from pandas._libs.missing import NAType

from resample.constants import (
    HPO_DICT_BASE_KEYS, 
    PRED_DATAFRAME_RESULTS_FIXED_COLUMNS
)



def get_estimator_filepath(pars: dict, repeat: int | NAType, fold: int) -> str:
    '''Get the filename to save the estimator'''
    resample_iteration_signature = get_resample_iteration_signature(repeat, fold)
    filename = f"{pars["estimator"]}__{pars["preprocessing"]}__{resample_iteration_signature}.pkl"
    return f"{pars["output_dir"]}/estimators/{filename}"



def get_resample_iteration_signature(repeat: int | NAType, fold: int) -> str:
    '''
    Get the iteration signature as string, that is: 
    - {repeat}{fold} in cross-validation
    - {fold} in resample
    '''
    if pd.isna(repeat):
        return f"{repeat}{fold}"
    else:
        return f"{fold}"



def create_dict_hpo(pars: dict) -> dict[str, list]:
    '''
    Creates the dictionary used to store the tuning info.
    The dict is empy when tuning is not requested.
    Note: Now it needs the hps dict in input.
    '''
    if not pars["tune"]: return {}
    hpo_params_keys = list(pars["tune_configuration"]["params_distributions"].keys())
    loss_keys = [f"loss_{i}" for i in range(pars["tune_configuration"]["n_iter"])]
    return {
        key: [] 
        for key in HPO_DICT_BASE_KEYS + hpo_params_keys + loss_keys
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