from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Callable, Any, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metalearning.sampler import HyperoptRandomSampler
    from metalearning.metafeatures import CustomMFE
    



class SurrogateWorker:
    '''
    Class that integrates the hp sampler, metafeature extractor, surrogate framework, 
    and acquisition functions to manage the generation and evaluation of meta-points. 
    It supports various strategies (propose_* methods) for selecting the meta-points.

    Parameters:
        sampler (HyperoptRandomSampler):
            Sampler that allows to sample hp points from a space.
        mfe (CustomMFE):
            CustomMFE to extract data metafeatures.
        surrogate_framework (Pipeline): 
            Fitted suggorate framework to infer the quality of meta-points.
        acquisition_func (Callable): 
            Acquisiton function to evaluate the promisingness of meta-points.
    '''
    def __init__(
        self,
        sampler: HyperoptRandomSampler,
        mfe: CustomMFE,
        surrogate_framework: Pipeline,
        acquisition_func: Callable[[Any], np.ndarray]
    ):
        self.sampler=sampler
        self.mfe=mfe
        self.surrogate_framework=surrogate_framework
        self.acquisition_func=acquisition_func

    
    def fit(
        self, 
        X: pd.DataFrame | np.ndarray, 
        y: pd.Series | np.ndarray, 
        hp_space: dict,
        seed: int
    ) -> "SurrogateWorker":
        '''
        Initialize the worker with the data, hyperparameter space, and random seed.
        The provided `hp_space` must be compatible with the assigned sampler.

        Parameters:
            X (pd.DataFrame | np.ndarray): Feature matrix.
            y (pd.Series | np.ndarray): Target vector.
            hp_space (dict): Hyperparameter space.
            seed (int): Random seed controlling candidate sampling.

        Returns:
            SurrogateWorker: The fitted instance.
        '''
        self.X=X
        self.y=y
        self.hp_space=hp_space
        self.seed=seed
        self.is_fitted_=True
        return self


    def propose_n_best(
        self, 
        n_candidate_points: int, 
        n_best: int,
        mfe_fit_kwargs: None | dict = None,
        mfe_extract_kwargs: None | dict = None,
        sampler_kwargs: None | dict = None,
        acquisition_func_kwargs: None | dict = None
    ) -> list[dict[str, Any]]:
        '''
        Get the best meta-points. 
        Here by best we mean the ones that maximize the promisingness score 
        evaluated though the surrogate framework and acquisition function.

        Parameters:
            n_candidate_points (int): 
                Number of points to draw as candidates.

            n_best (int): 
                Number of points returned by the utility.
            
            mfe_fit_kwargs (None | dict):
                Kwargs to pass to the mfe `fit` method.
            
            mfe_extract_kwargs (None | dict):
                Kwargs to pass to the mfe `extract` method.

            sampler_kwargs (None | dict):
                Kwargs to pass to the sampler `sample_points` method.

            acquisition_func_kwargs (None | dict):
                Kwargs to pass to the acquisition function callable.

        Returns:
            list[dict[str,Any]]: The list of the best points.
        '''
        check_is_fitted(self, "is_fitted_")
        acquisition_func_kwargs = ensure_or_create(acquisition_func_kwargs, dict)

        metadata, candidate_points = self._generate_meta_data(
            n_candidate_points=n_candidate_points, 
            mfe_fit_kwargs=mfe_fit_kwargs,
            mfe_extract_kwargs=mfe_extract_kwargs,
            sampler_kwargs=sampler_kwargs
        )
       
        pred_values, pred_uncertainty = self.surrogate_framework.predict(metadata)        
        promisingness = self.acquisition_func(pred_values, pred_uncertainty, **acquisition_func_kwargs)

        # argsort works in the increasing order (last index --> index of the greatest value)
        top_idx = np.argsort(promisingness, stable=True)[-n_best:]
        selected_points = [candidate_points[idx] for idx in top_idx]
        return selected_points


    def _generate_meta_data(
        self,
        n_candidate_points: int, 
        mfe_fit_kwargs: None | dict,
        mfe_extract_kwargs: None | dict,
        sampler_kwargs: None | dict
    ) -> tuple[pd.DataFrame, list[dict]]:
        '''
        Generate the meta-data, i.e. sampled hps + data metafeatures.
        Returns the meta-data plus the list of candidate hp points used to build it.
        Importantly the meta-data and candidate points order matches, meaning
        that the first row is built upon the first point in the list and so on.
        '''
        mfe_extract_kwargs = ensure_or_create(mfe_extract_kwargs, dict)
        mfe_fit_kwargs = ensure_or_create(mfe_fit_kwargs, dict)
        sampler_kwargs = ensure_or_create(sampler_kwargs, dict)

        candidate_points = self.sampler.fit(self.hp_space, self.seed).sample_points(
            n_candidate_points, 
            **sampler_kwargs
        )
        
        df_candidate_points = pd.DataFrame(candidate_points)
        metafeatures = self.mfe.fit(self.X, self.y, **mfe_fit_kwargs).extract(**mfe_extract_kwargs)
        
        # we create a copy since the original df is not optimized in memory due to assign
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)
            df_candidate_points = df_candidate_points.assign(**metafeatures).copy()

        return df_candidate_points, candidate_points