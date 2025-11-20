from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metalearning.metadata_generator import MetadataGenerator
    from metatab_utils.types import XType, YType



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
        X: XType, 
        y: YType, 
        hp_space: dict,
        seed: int
    ) -> "SurrogateWorker":
        '''
        Initialize the worker with the data, hyperparameter space, and random seed.
        The provided `hp_space` must be compatible with the assigned sampler.

        Parameters:
            X (XType): Feature matrix.
            y (YType): Target vector.
            hp_space (dict): Hyperparameter space.
            seed (int): Random seed controlling candidate sampling.

        Returns:
            SurrogateWorker: The fitted instance.
        '''
        self.metadata_generator.fit(X, y, hp_space, seed)
        self.is_fitted_=True
        return self


    def draw_candidates(
        self,
        n_candidate_points: int, 
        point_corrector_kwargs: None | dict = None,
        mfe_fit_kwargs: None | dict = None,
        mfe_extract_kwargs: None | dict = None
    ) -> tuple[pd.DataFrame, list[dict]]:
        '''
        Draw candidate points.
        These are both set internally and returned.

        Parameters:
            n_candidate_points (int): 
                Number of points to draw as candidates.
            
            point_corrector_kwargs (None | dict, optional):
                Kwargs to pass to the PointCorrector `correct_point` method.

            mfe_fit_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `fit` method.
            
            mfe_extract_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `extract` method.

        Returns:
            tuple[pd.DataFrame,list[dict]]:
            Returns the meta-data plus the list of hp points used to build it.
            Importantly the meta-data and points order matches, meaning
            that the first row is built upon the first point in the list and so on.
        '''
        check_is_fitted(self, "is_fitted_")
        metadata, candidate_points = self.metadata_generator.generate(
            n_points=n_candidate_points,
            point_corrector_kwargs=point_corrector_kwargs,
            mfe_fit_kwargs=mfe_fit_kwargs,
            mfe_extract_kwargs=mfe_extract_kwargs
        )
        self.n_candidate_points_ = n_candidate_points
        self.metadata_ = metadata
        self.candidate_points_ = candidate_points
        return metadata, candidate_points
       

    def evaluate_candidates(self, acquisition_func_kwargs: None | dict = None) -> np.ndarray:
        '''
        Evaluate the promisingness of the drawn candidate points,
        through the surrogate framework and acquisition function.
        This info is both returned and set internally.

        Parameters:
            acquisition_func_kwargs (None | dict):
                Kwargs to pass to the acquisition function callable.

        Returns:
            np.ndarray: The candidate points promisingness scores.
        '''
        self._check_candidate_points_existence()
        pred_values, pred_uncertainty = self.surrogate_framework.predict(self.metadata_)        
        acquisition_func_kwargs = ensure_or_create(acquisition_func_kwargs, dict)
        promisingness = self.acquisition_func(pred_values, pred_uncertainty, **acquisition_func_kwargs)
        self.promisingness_ = promisingness
        return promisingness
    

    def propose_n_best(self, n_best: int) -> list[dict[str, Any]]:
        '''
        Get the best meta-points. 
        Here by best we mean the ones that maximize the promisingness score.

        Parameters:
            n_best (int): 
                Number of points returned by the utility.

        Returns:
            list[dict[str,Any]]: The list of best points.
        '''
        self._check_candidate_points_existence()
        self._check_promisingness_score_existence()

        if n_best > self.n_candidate_points_:
            raise ValueError("'n_best' cannot be greater than the number of candidate points.")

        # argsort works in increasing order, so we reverse to have best --> worst direction
        top_idx = np.argsort(self.promisingness_, stable=True)[::-1][:n_best]
        return [self.candidate_points_[idx] for idx in top_idx]
    

    def propose_random_from_top(
        self, 
        n_proposed: int,
        top: int,
        seed: int = 0
    ) -> list[dict[str, Any]]:
        '''
        Get the best points from the top positions randomly without duplication.
        This propose strategy let to randomly draw `n_proposed` points from the `top`
        positions of `n_candidate_points` in a random way without duplication.

        Parameters:
            n_proposed (int): 
                Number of points returned by the utility.

            top (int):
                Number of top points from which draw the random proposed ones.
            
            seed (int, optional):
                Control the randomness of the point selection procedure.   

        Returns:
            list[dict[str,Any]]: The list of best points.
        '''
        self._check_candidate_points_existence()
        self._check_promisingness_score_existence()

        if top > self.n_candidate_points_:
            raise ValueError(f"'top' cannot be greater than the number of candidate points.")

        if n_proposed > top:
            raise ValueError(f"'n_proposed' cannot be greater than 'top'.")

        # argsort works in increasing order, so we reverse to have best --> worst direction
        top_idx = np.argsort(self.promisingness_, stable=True)[::-1][:top]
        top_points = [self.candidate_points_[idx] for idx in top_idx]

        # we can skip random drawing in this scenario
        if top == n_proposed:
            return top_points
        
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(top, size=n_proposed, replace=False)
        return [top_points[idx] for idx in selected_idx]


    def propose_uniform_from_top(self, n_steps: int, step_size: int):
        '''
        Get the best `n_steps` points using a fixed `step_size`.
        This propose strategy can be think of ordering the points
        in the best --> worst direction, and then selecting an item
        every step_size elements, for n_steps times. 
        The utility therefore returns n_steps points.

        Parameters:
            n_steps (int): 
                Number of points returned by the utility.

            step_size (int):
                Size of the step.

        Returns:
            list[dict[str,Any]]: The list of the best points.
        '''
        self._check_candidate_points_existence()
        self._check_promisingness_score_existence()

        working_range = n_steps * step_size
        
        if working_range > self.n_candidate_points_:
            raise ValueError(
                "'n_steps * step_size' cannot be greater than the number of candidate points."
            )

        # argsort works in increasing order, so we reverse to have best --> worst direction
        ordered_idx = np.argsort(self.promisingness_, stable=True)[::-1]
        ordered_points = [self.candidate_points_[idx] for idx in ordered_idx]
        return ordered_points[:working_range:step_size]


    def _check_candidate_points_existence(self) -> None:
        if not hasattr(self, "candidate_points_"):
            raise ValueError(
                "There are no candidates to evaluate. You must call 'draw_candidates' method first."
            ) 
    
    def _check_promisingness_score_existence(self) -> None:
        if not hasattr(self, "promisingness_"):
            raise ValueError(
                "The candidate points promisingness score has not been evaluated yet."
            )