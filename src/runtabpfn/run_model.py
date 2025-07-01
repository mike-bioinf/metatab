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
from finetabpfn import AesFineTunedTabPFNClassifier
from runtabpfn.constants import Classifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier



def get_repetition_fold(iteration: int, pars: dict) -> tuple: 
    '''
    Utility to get the repetition and fold info based on 
    the current iteration and splitting mode.
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

    if model == "base":
        return TabPFNClassifier(**model_specs)  
    elif model == "auto":
        return AutoTabPFNClassifier(**model_specs)  
    elif model == "rf":
        return  RandomForestClassifier(**model_specs)
    elif model == "aes_ft":
        return AesFineTunedTabPFNClassifier(**model_specs)
    else:
        raise NotImplementedError(f"'{model}' must be one of: 'base', 'auto', 'rf' and 'aes_ft'.")



def create_estimator(
    classifier: Classifier, 
    type_preprocessing: Literal["no", "filter", "pca"],
    pars: dict,
) -> Classifier | Pipeline | GridSearchCV:
    '''
    Utility to get the final estimator according to the scenario.
    Note: in grid search cv uses a fixed 5-repeated 5-fold cv and "roc_auc" as scoring metric.
    Returns a Classifier, Pipeline or GridSearchCV object.
    '''
    grid_search = pars["grid_search"]
    seed = pars["seed"]
    is_rf_clf = isinstance(classifier, RandomForestClassifier)

    # in case of pca preprocessing the trasformation steps are identical for all classifiers
    if type_preprocessing == "pca":
        estimator = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(random_state=0), classifier)
    elif is_rf_clf:
        estimator = make_pipeline(VarianceThreshold(), StandardScaler(), classifier)
    else:
        estimator = classifier

    if grid_search is not None:
        estimator = GridSearchCV(
            estimator=estimator, 
            param_grid=grid_search, 
            scoring="roc_auc", 
            n_jobs=-1, 
            refit=True, 
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        )
    
    return estimator



def universal_predict_proba(
    estimator: Classifier | Pipeline | GridSearchCV, 
    X: np.ndarray | pd.DataFrame, 
    **kwargs
) ->  np.ndarray:
    '''
    Adapter that invokes the estimators predict_proba method
    allowing for additional parameters. 
    Relies on the assumption that the estimator predict methods 
    accept the X as first argument.
    Necessary since AesFineTunedTabPFNClassifier does not 
    respect the usual predicts signature.

    Parameters:
        estimator (Classifier | Pipeline | GridSearchCV):
            The estimator object that calls the predict_proba method.
        X (np.ndarray | pd.DataFrame):
            The test data.
        kwargs:
            Other keyword arguments.
    
    Returns:
        np.ndarray: The predicted probabilities.
    '''
    if isinstance(estimator, AesFineTunedTabPFNClassifier):
        return estimator.predict_proba(X, **kwargs)
    else:
        return estimator.predict_proba(X)