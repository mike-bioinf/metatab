import pandas as pd
from functools import partial
from typing import Any
from sklearn.model_selection import GridSearchCV
from finetabpfn import SklearnFineTuneTabPFN, HPS_FINETUNE
from runtabpfn.constants import PRED_DATAFRAME_ADDITIONAL_COLUMNS, HPO_DICT_BASE_KEYS, Classifier



def create_dict_hpo(pars: dict, base_keys: list[str] = HPO_DICT_BASE_KEYS) -> dict[str, list]:
    '''
    Creates the dictionary used to store the best hyperparameters info.
    This dict is used either when we use the random forest with hpo, or when we finetune tabpfn with hpo.
    Note: returns an empty dict when no hpo is involved.
    '''
    model = pars["model"] 
    grid_search = pars["grid_search"] 
    d = {}

    is_optimization_involved = (model == "rf" and grid_search is not None) or model == "ft_opt"

    if not is_optimization_involved:
        return d
    
    # add the base training keys
    for key in base_keys:
        d.update({key: []})
    
    # add the hpo scenario-specific keys
    hpo_specific_keys = list(grid_search.keys()) if grid_search is not None else HPS_FINETUNE
    
    for key in hpo_specific_keys:
        d.update({key: []})
        
    return d



def populate_dict_hpo_(
        dict_hpo: dict[str, list],
        model: str, 
        classifier: Classifier,
        splitting_mode: str,        
        preprocessing: str,
        repetition: int,
        fold: int
    ) -> None:
    '''Populate the HPO dict in place'''
    if isinstance(classifier, GridSearchCV) or (isinstance(classifier, SklearnFineTuneTabPFN) and model == "ft_opt"):
        dict_hpo["splitting_mode"].append(splitting_mode)
        dict_hpo["preprocessing"].append(preprocessing)
        dict_hpo["repetition"].append(repetition)
        dict_hpo["fold"].append(fold)

        hpo_instance = classifier if isinstance(classifier, GridSearchCV) else classifier.study_
        best_params_attr = "best_params_" if isinstance(classifier, GridSearchCV) else "best_params"
        
        for key, value in getattr(hpo_instance, best_params_attr).items():
            dict_hpo[key].append(value)
        


def create_dict_results(additional_columns: list[str] = PRED_DATAFRAME_ADDITIONAL_COLUMNS) -> dict:
    '''Utility to create the dict result'''
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
    # updating model_specs with the ft_wrapper_specs previsously separated
    model_specs = pars["model_specs"]
    model_specs.update(pars["ft_wrapper_specs"])
    
    return {
        "input_path": str(pars["input_path"]),
        "output_path": str(pars["output_path"]),
        "input_mode": pars["input_mode"],
        "splitting_mode": pars["splitting_mode"],
        "splitting_specs": pars["splitting_specs"],
        "model": pars["model"],
        "model_specs": secure_str(model_specs, 'not serializable'),
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