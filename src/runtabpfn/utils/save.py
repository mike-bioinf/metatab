import pandas as pd
from functools import partial
from typing import Any
from sklearn.model_selection import GridSearchCV
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from runtabpfn.utils.constants import ADDITIONAL_COLUMNS, HPO_DICT_KEYS



def create_dict_hpo(grid_search: None | dict, hpo_dict_keys: list[str] = HPO_DICT_KEYS) -> dict[str, list]:
    d = {}
    for key in hpo_dict_keys:
        d.update({key: []})
    if grid_search is not None:
        for key in grid_search.keys():
            d.update({key: []})
    return d



def populate_dict_hpo_(
        dict_hpo: dict[str, list],
        classifier: GridSearchCV | TabPFNClassifier | AutoTabPFNClassifier,
        splitting_mode: str,        
        preprocessing: str,
        repetition: int,
        fold: int
    ) -> None:
    if isinstance(classifier, GridSearchCV):
        dict_hpo["splitting_mode"].append(splitting_mode)
        dict_hpo["preprocessing"].append(preprocessing)
        dict_hpo["repetition"].append(repetition)
        dict_hpo["fold"].append(fold)
        for key, value in classifier.best_params_.items():
            dict_hpo[key].append(value)



def create_dict_results(additional_columns: list[str] = ADDITIONAL_COLUMNS) -> dict:
    '''Utility to create a base dict result'''
    d = {"dataset": [], "y_train": [], "y_test": [], "pred_proba": []}
    for key in additional_columns:
        d.update({key: []})
    return d



def _populate_dict_result_(dict_results: dict[str, list], **kwargs) -> None:
    '''Utility to extend the dict results internal lists. Modifies the dict in place.'''
    for key, value in kwargs.items():
        dict_results[key].append(value)



populate_dict_result_ = partial(
    _populate_dict_result_,
    number_initial_features=pd.NA,  
    number_filtered_features=pd.NA,
    filtering_threshold=pd.NA,
    number_pca_components=pd.NA
)



def create_configuration_dict(pars: dict) -> dict:
    return {
        "input_path": str(pars["input_path"]),
        "output_path": str(pars["output_path"]),
        "input_mode": pars["input_mode"],
        "splitting_mode": pars["splitting_mode"],
        "splitting_specs": pars["splitting_specs"],
        "model": pars["model"],
        "model_specs": secure_str(pars["model_specs"], 'not serializable'),
        "grid_search": secure_str(pars["grid_search"], 'not serializable'),
        "preprocessing": pars["preprocessing"],
        "test_dataset": pars["test_dataset"],
        "target_feature": pars["target_feature"],
        "seed": pars["seed"]
    }



def secure_str(obj: Any, exception_string: str, passnone: bool = False) -> str | None:
    '''
    Safely returns the string representation of the object.
    Parameters:
        obj (Any): object on which str is called.
        exception_string (str): String returned if str(obj) raises an expection.
        passnone (bool, optional): Whether to pass None as is and not the string representation 'None'.
    Returns:
        str|None: Either str(obj), exception_string or None.
    '''
    if obj is None and passnone:
        return obj
    
    try:
        str_obj = str(obj)
    except Exception:
        str_obj = exception_string
    
    return str_obj
