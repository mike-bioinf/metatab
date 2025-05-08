import numpy as np
import pandas as pd
from typing import Generator, Literal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from finetabpfn import SklearnFineTuneTabPFN, FineTuneTabPFN
from runtabpfn.constants import Classifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier



class MockSplitter:
    '''Class that mimick a sklearn splitter with a single yield of a binary tuple of np.nan'''
    def __init__(self):
        pass

    def split(self, *args, **kwargs) -> Generator[tuple, None, None]:
        yield (np.nan, np.nan)



def pick_splitter(pars: dict):
    '''Utility to pick and create the right splitter depending on splitting_mode'''
    splitting_mode = pars["splitting_mode"]
    splitting_specs = pars["splitting_specs"]
    seed = pars["seed"]
    
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



def pick_classifier(pars) -> Classifier:
    '''
    Utility to select and build the right classifier.
    Returns the classifier.
    ''' 
    model = pars["model"]
    model_specs = pars["model_specs"]
    ft_wrapper_specs = pars["ft_wrapper_specs"]
    seed = pars["seed"]

    if model in ["auto", "rf"]:
        model_specs["random_state"] = np.random.RandomState(seed)

    if model in ["ft", "ft_opt"]:
        model_specs["random_state"] = seed

    if model == "base":
        return TabPFNClassifier(**model_specs)  
    elif model == "auto":
        return AutoTabPFNClassifier(**model_specs)  
    elif model == "rf":
        return  RandomForestClassifier(**model_specs)
    elif model in ["ft", "ft_opt"]:
        return SklearnFineTuneTabPFN(FineTuneTabPFN(**model_specs), **ft_wrapper_specs)
    else:
        raise NotImplementedError(f"'{model}' must be one of: 'base', 'auto', 'rf', 'ft' or 'ft_opt'.")



def create_classifier_pipeline(
        classifier: Classifier, 
        type_preprocessing: Literal["no", "filter", "pca"],
        pars: dict,
    ) -> Classifier | Pipeline | GridSearchCV:
    '''
    Utility to insert the classifier in the correct pipeline according to the scenario.
    Note: Uses a fixed 5-repeated 5-fold cv for random forest grid search HPO.
    Returns a Classifier, Pipeline or GridSearchCV object.
    '''
    grid_search = pars["grid_search"]
    seed = pars["seed"]
    is_rf_clf = isinstance(classifier, RandomForestClassifier)

    # in case of pca preprocessing the trasformation steps are identical for all classifiers
    if type_preprocessing == "pca":
        pipe = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(random_state=0), classifier)
    elif is_rf_clf:
        pipe = make_pipeline(VarianceThreshold(), StandardScaler(), classifier)
    else:
        pipe = classifier

    if grid_search is not None:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        pipe = GridSearchCV(pipe, param_grid=grid_search, scoring="roc_auc", n_jobs=-1, refit=True, cv=cv)
    
    return pipe



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