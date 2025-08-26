from __future__ import annotations

import pickle
from copy import deepcopy
from typing import Literal, TYPE_CHECKING
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from estimators.searchcv import SearchCV

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from preprocessing.density_selector import DensityFeatureSelector
    from estimators.types import Classifier



class AbstractBaseEstimator(ABC):
    '''
    Abstract base class for estimators classes.
    
    The estimators classes must implement the 'estimator_' attribute
    learned in the 'fit' method, storing the object fitted on the input data.
    This can be a Classifier, Pipeline or SearchCV object.
    
    Note: fixed_params must always be optional i.e. it must have a default.
    
    Parameters:
        preprocessing (Literal["base", "density_filter", "pca"]): 
            Preprocessing strategy to use.
        
        seed (int): 
            Seed for reproducibility.
            It is used to fit the estimator and for
            eventual splitting and tuning procedures. 

        n_threads (int):
            Number of CPU threads used to fit the estimator. 
            Is ignored by not parallelizable estimators.

        early_stopping_rounds (int):
            Number of eraly stop rounds used by the early stopped estimators.
            The value is ignored by the non early stopped estimators.

        tune_configuration (None | dict):
            Dict with the tuning info. 
            Must be ignored by the not tunable estimators.
            
        fixed_params (dict, optional):
            Dict of param:value that are fixed.
    '''
    @abstractmethod
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_threads: int,
        early_stopping_rounds: int,
        tune_configuration: None | dict,
        fixed_params: dict
    ):
        self.preprocessing = preprocessing
        self.seed = seed
        self.n_threads = n_threads
        self.early_stopping_rounds = early_stopping_rounds
        self.tune_configuration = tune_configuration
        self.fixed_params = fixed_params
        
    
    @abstractmethod
    def fit(*args, **kwargs):
        pass
    

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Executes the "classic" predict_proba method 
        involving only the X parameter, 
        plus good defaults for the other ones.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


    def save(self, filepath: str | Path):
        '''Seriealize the instance using pickle'''
        if not hasattr(self, "estimator_"):
            warn(
                message="The estimator instance is not fitted! Is this expected?", 
                category=UserWarning
            )
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    

    # to override if needed by concrete classes
    def get_best_hps(self) -> dict | None:
        '''Get the best HPs resulting from tuning'''
        check_is_fitted(self, "estimator_")
        if isinstance(self.estimator_, SearchCV):
            return self.estimator_.best_params_

    
    # to override if needed by concrete classes
    def get_search_losses(self) -> np.ndarray | None:
        '''Get the search losses resulting from tuning'''
        check_is_fitted(self, "estimator_")
        if isinstance(self.estimator_, SearchCV):
            return np.array(self.estimator_.trials_.losses())
    

    def update_fixed_params(
        self,
        *,
        up_seed: bool = False, 
        up_n_threads: bool = False,
        up_early_stopping_rounds: bool = False, 
        key_seed: str = "random_state", 
        key_n_threads: str = "n_jobs",
        key_early_stopping_rounds: str = "early_stopping_rounds",
        copy: bool = False
    ) -> dict:
        '''
        Update the fixed params dict or a deepcopy of it 
        with the seed, n_threads and early_stopping_rounds info.
        Returns the updated dict.
        '''
        fixed_params = deepcopy(self.fixed_params) if copy else self.fixed_params
        if up_seed: fixed_params[key_seed] = self.seed
        if up_n_threads: fixed_params[key_n_threads] = self.n_threads
        if up_early_stopping_rounds: fixed_params[key_early_stopping_rounds] = self.early_stopping_rounds
        return fixed_params
                

    def get_feature_names_in_(self) -> np.ndarray:
        '''Returns the "feature_names_in_" attribute learned at fit level'''
        check_is_fitted(self, "estimator_")
        fitted_obj = self._retrieve_fitted_obj()
        return fitted_obj.feature_names_in_


    def collect_fit_preprocessing_info(self) -> dict:
        '''
        Collect the learned preprocessing attributes of interest in a dict.
        An empty dict is returned in case no preprocessing is done.
        '''
        check_is_fitted(self, "estimator_")
        fitted_obj = self._retrieve_fitted_obj()
        if isinstance(fitted_obj, Pipeline): 
            return self._collect_fit_preprocessing_info(fitted_obj)
        else:
            return {}


    def _retrieve_fitted_obj(self) -> Classifier | Pipeline:
        '''Retrieve the fitted object, i.e. the classifier or the pipeline.'''
        fitted_obj = self.estimator_.best_estimator_ \
            if isinstance(self.estimator_, SearchCV)\
            else self.estimator_
        return fitted_obj


    def _collect_fit_preprocessing_info(self, pipeline: Pipeline) -> dict:
        '''Internal to collect the preprocessing info from the fitted Pipeline'''
        if self.preprocessing == "pca":
            return self._collect_from_pca_preprocessing(pipeline)
        elif self.preprocessing == "density_filter":
            return self._collect_from_density_preprocessing(pipeline)
        elif self.preprocessing == "base":
            return {}

    
    def _collect_from_pca_preprocessing(self, pipeline: Pipeline) -> dict:
        '''Collect the pca related learned info'''
        pca: PCA = pipeline.named_steps["pca"]
        # we wrap the container objects to avoid errors 
        # in the building process of the prediction dataframe object
        return {
            "n_pca_components": pca.n_components_,
            "explained_variance_ratio": [pca.explained_variance_ratio_],
            "total_explained_variance_ratio": pca.explained_variance_ratio_.sum()
        }
    
    
    def _collect_from_density_preprocessing(self, pipeline: Pipeline) -> dict:
        '''Collect the density related learned info'''
        density_selector: DensityFeatureSelector = pipeline.named_steps["densityfeatureselector"]
        return {
            "density_selection_strategy": density_selector.strategy_,
            "n_target_features": density_selector.n_target_features_,
            "minimum_selected_density_score": density_selector.minimum_density_score_
        }