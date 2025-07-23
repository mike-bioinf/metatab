from __future__ import annotations

import pickle
from copy import deepcopy
from typing import Literal, TYPE_CHECKING
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from estimators.preprocessing.density_selector import DensityFeatureSelector
    from estimators.estimators.types import TabPFNEstimators



class AbstractBaseEstimator(ABC):
    '''
    Abstract and Base class for estimators classes.
    
    The estimators classes must implement the 'estimator_' attribute
    in the "fit" method, storing the model fitted on the input data.
    
    Note: the params_distributions and fixed_params must always be optional,
    i.e. they must implement defaults.
        
    Parameters:
        preprocessing (Literal["base", "density_filter", "pca"]): 
            Preprocessing strategy to use.
        
        seed (int): 
            Seed for reproducibility.
            This seed is directly used to fit the model.
            It is used also for eventual splitting and tune procedures. 

        n_cores (int):
            Number of CPU cores used to fit the estimator. 
            Is ignored by the unparallelizable estimators.

        params_distributions (dict | None, optional):
            Dict of param:distributions from which to sample values in the tuning process.
            Can be None.
        
        fixed_params (dict, optional):
            Dict of param:value that are fixed i.e. not tuned in the search.

    '''
    @abstractmethod
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_cores: int,
        params_distributions: dict | None,
        fixed_params: dict
    ):
        self.preprocessing = preprocessing
        self.seed = seed
        self.n_cores = n_cores
        self.params_distributions = params_distributions
        self.fixed_params = fixed_params
        
    
    @abstractmethod
    def fit(*args, **kwargs):
        pass
    

    @abstractmethod
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline | TabPFNEstimators:
        '''
        Return the fitted preprocessing pipeline 
        with or without the classifier head, or the fitted estimator
        in the absence of it. Tabpfn-derived estimators does 
        not require/build a sklearn pipeline with base preprocessing. 
        For these classes the estimator is returned.
        '''
        pass
    

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Executes the "classic" predict_proba framework and method, 
        i.e. without external/decoupled test data preprocessing 
        and with the classic method signature involving only the X parameter, 
        plus good defaults for other eventual ones.
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
    

    def get_best_hps(self) -> None:
        '''
        Get the best HPs resulting from tuning.
        Returns None since this method is the one 
        used by the estimators that does not tune HPs.
        '''
        return None


    def update_fixed_params(
        self,
        *,
        up_seed: bool, 
        up_n_cores: bool, 
        key_seed: str = "random_state", 
        key_n_cores: str = "n_jobs", 
        copy: bool = False
    ) -> dict:
        '''
        Update the fixed params dict or a deepcopy of it 
        with the seed and n_cores info.
        Returns the updated dict.
        '''
        fixed_params = deepcopy(self.fixed_params) if copy else self.fixed_params
        if up_seed: fixed_params[key_seed] = self.seed
        if up_n_cores: fixed_params[key_n_cores] = self.n_cores
        return fixed_params
                

    def get_feature_names_in_(self) -> np.ndarray:
        '''
        Returns the "feature_names_in" attribute learned at fit level.
        The attribute is retrieved from the fitted preprocessing pipeline 
        or from the estimator in absence of the first.
        '''
        check_is_fitted(self, "estimator_")
        fitted_obj = self._get_fitted_preprocessing_pipeline_or_estimator()
        return fitted_obj.feature_names_in_


    def collect_fit_preprocessing_info(self) -> dict:
        '''
        Collect the learned preprocessing attributes of interest in a dict.
        An empty dict is returned in case the estimator has no preprocessing pipeline.
        '''
        check_is_fitted(self, "estimator_")
        fitted_obj = self._get_fitted_preprocessing_pipeline_or_estimator()
        if not isinstance(fitted_obj, Pipeline):
            return {}
        return self._collect_fit_preprocessing_info(fitted_obj)


    def _collect_fit_preprocessing_info(self, preprocessing_pipeline: Pipeline) -> dict:
        '''
        Internal to collect the preprocessing info.
        Wants in input the preprocessing pipeline.
        This means either the sole decoupled preprocessing pipeline, 
        like for ESXGB estimators, or the pipeline headed by the classifier.
        '''
        if self.preprocessing == "pca":
            return self._collect_from_pca_preprocessing(preprocessing_pipeline)
        elif self.preprocessing == "density_filter":
            return self._collect_from_density_preprocessing(preprocessing_pipeline)
        elif self.preprocessing == "base":
            return {}

    
    def _collect_from_pca_preprocessing(self, preprocessing_pipeline: Pipeline) -> dict:
        '''Collect the pca related learned info'''
        pca: PCA = preprocessing_pipeline.named_steps["pca"]
        # we wrap the container objects to avoid errors 
        # in the building process of the prediction dataframe object
        return {
            "n_pca_components": pca.n_components_,
            "explained_variance_ratio": [pca.explained_variance_ratio_],
            "total_explained_variance_ratio": pca.explained_variance_ratio_.sum()
        }
    
    
    def _collect_from_density_preprocessing(self, preprocessing_pipeline: Pipeline) -> dict:
        '''Collect the density related learned info'''
        density_selector: DensityFeatureSelector = preprocessing_pipeline.named_steps["densityfeatureselector"]
        return {
            "density_selection_strategy": density_selector.strategy_,
            "n_target_features": density_selector.n_target_features_,
            "minimum_selected_density_score": density_selector.minimum_density_score_
        }