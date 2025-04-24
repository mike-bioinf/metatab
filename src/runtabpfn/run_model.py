import logging
import numpy as np
import pandas as pd
from typing import Generator, Any, Literal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier



class MockSplitter:
    '''Class that mimick a sklearn splitter with a single yield of a binary tuple of np.nan'''
    def __init__(self):
        pass

    def split(self, *args, **kwargs) -> Generator[tuple, None, None]:
        yield (np.nan, np.nan)



def pick_splitter(splitting_mode: str, splitting_specs: dict, seed: int):
    '''Utility to pick and create the right splitter depending on splitting_mode'''
    if splitting_mode == "cv":
        splitter = RepeatedStratifiedKFold(
            n_splits=splitting_specs["n_splits"], 
            n_repeats=splitting_specs["n_repeats"], 
            random_state=seed
            )
    elif splitting_mode == "holdout":
        splitter = StratifiedShuffleSplit(
            n_splits=splitting_specs["n_splits"], 
            train_size=splitting_specs["train_size"], 
            random_state=seed
        )
    else:
        splitter = MockSplitter()
    
    return splitter



def pick_model(model: str, model_specs: dict, grid_search: None | dict, seed: int):
    '''
    Utility to select the right model. 
    Sets the right preprocessing pipeline in case of "rf". The pipeline is fixed.
    Sets also the cv specs if HPO is desired in case of "rf". Also these are fixed.  
    ''' 
    if model in ["auto", "rf"]:
        model_specs["random_state"] = np.random.RandomState(seed)

    if model == "base":
        return TabPFNClassifier(**model_specs)  
    elif model == "auto":
        return AutoTabPFNClassifier(**model_specs)  
    elif model == "rf" and grid_search is None:
        return  RandomForestClassifier(**model_specs)
    elif model == "rf" and grid_search is not None:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        return GridSearchCV(RandomForestClassifier(**model_specs), param_grid=grid_search, scoring="roc_auc", n_jobs=-1, refit=True, cv=cv)
    else:
        raise NotImplementedError(f"'{model}' must be one of: 'base', 'auto' or 'rf'.")



def create_classifier_pipeline(classifier: Any, type_preprocessing: Literal["no", "filter", "pca"]) -> Any | Pipeline:
    '''
    Utility to create the transformation pipeline with the classifier as last step.
    Returns a Pipeline object or the classifier depending on the setting.
    '''
    is_tabpfn_clf = isinstance(classifier, (TabPFNClassifier, AutoTabPFNClassifier))
    
    if is_tabpfn_clf and type_preprocessing != "pca":
        return classifier
    elif not is_tabpfn_clf and type_preprocessing != "pca":
        return make_pipeline(VarianceThreshold(), StandardScaler(), classifier)
    elif type_preprocessing == "pca":
        return make_pipeline(VarianceThreshold(), StandardScaler(), PCA(), classifier)



def get_repetition_fold(iteration: int, pars: dict) -> tuple: 
    '''
    Utility to get the repetition and fold info based on the current iteration and splitting mode.
    Returns a binary tuple of int and/or pd.NA.
    '''
    if pars["splitting_mode"] == "cv":
        repetition = iteration // pars["splitting_specs"]["n_splits"]
        fold = iteration - (pars["splitting_specs"]["n_splits"] * repetition)
    elif pars["splitting_mode"] == "holdout":
        fold, repetition = iteration, pd.NA
    else:
        fold, repetition = pd.NA, pd.NA
    
    return repetition, fold
