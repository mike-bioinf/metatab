from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metalearning.generator import MetadataGenerator



class SurrogateWorker:
    '''
    Class that integrates the metadata generator, surrogate framework, 
    and acquisition functions to manage the generation and evaluation of meta-points. 
    It supports various strategies (propose_* methods) for selecting the meta-points.

    Parameters:
        metadata_generator (MetadataGenerator):
            Instance to generate metadata.
        surrogate_framework (Pipeline): 
            Fitted suggorate framework to infer the quality of meta-points.
        acquisition_func (Callable): 
            Acquisiton function to evaluate the promisingness of meta-points.
    '''
    def __init__(
        self,
        metadata_generator: MetadataGenerator,
        surrogate_framework: Pipeline,
        acquisition_func: Callable[[Any], np.ndarray]
    ):
        self.metadata_generator=metadata_generator
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
        self.metadata_generator.fit(X, y, hp_space, seed)
        self.is_fitted_=True
        return self


    def propose_n_best(
        self, 
        n_candidate_points: int, 
        n_best: int,
        point_corrector_kwargs: None | dict = None,
        mfe_fit_kwargs: None | dict = None,
        mfe_extract_kwargs: None | dict = None,
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
            
            point_corrector_kwargs (None | dict, optional):
                Kwargs to pass to the PointCorrector `correct_point` method.

            mfe_fit_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `fit` method.
            
            mfe_extract_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `extract` method.
            
            acquisition_func_kwargs (None | dict):
                Kwargs to pass to the acquisition function callable.

        Returns:
            list[dict[str,Any]]: The list of the best points.
        '''
        check_is_fitted(self, "is_fitted_")
        acquisition_func_kwargs = ensure_or_create(acquisition_func_kwargs, dict)

        metadata, candidate_points = self.metadata_generator.generate(
            n_points=n_candidate_points,
            point_corrector_kwargs=point_corrector_kwargs,
            mfe_fit_kwargs=mfe_fit_kwargs,
            mfe_extract_kwargs=mfe_extract_kwargs
        )
       
        pred_values, pred_uncertainty = self.surrogate_framework.predict(metadata)        
        promisingness = self.acquisition_func(pred_values, pred_uncertainty, **acquisition_func_kwargs)

        # argsort works in the increasing order (last index --> index of the greatest value)
        top_idx = np.argsort(promisingness, stable=True)[-n_best:]
        selected_points = [candidate_points[idx] for idx in top_idx]
        return selected_points