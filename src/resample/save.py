import json
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Literal
from pandas._libs.missing import NAType



def get_round_estimator_filepath(pars: dict, repeat: int | NAType, fold: int) -> str:
    '''Get the filename made of the estimator name and current round info'''
    resample_iteration_signature = get_resample_iteration_signature(repeat, fold)
    filename = f"{pars["estimator"]}_{resample_iteration_signature}.pkl"
    return f"{pars["output_dir"]}/estimators/{filename}"



def get_resample_iteration_signature(repeat: int | NAType, fold: int) -> str:
    '''
    Get the iteration signature as string, that is: 
    - {repeat}{fold} in cross-validation
    - {fold} in holdout
    '''
    if pd.isna(repeat):
        return f"{repeat}{fold}"
    else:
        return f"{fold}"



def populate_dict_lists_(dictionary: dict[str, list], **kwargs) -> None:
    '''Utility to extend in place the dictionary internal lists'''
    for key, value in kwargs.items():
        dictionary[key].append(value)



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

    if copy_pars["tune"]:
        del copy_pars["tune_configuration"]["params_distributions"]

    for key, value in copy_pars.items():
        if isinstance(value, Path):
            copy_pars[key] = str(value)
    
    with open(filepath, "w") as f:
        json.dump(copy_pars, f, indent=4)