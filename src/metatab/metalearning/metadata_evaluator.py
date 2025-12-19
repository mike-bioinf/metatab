from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline



class MetadataEvaluator:
    '''
    Class that evaluates metadata thorugh the surrogate model and acquistion function,
    and then proposes the best points according to different strategies (propose_* methods).

    Parameters:
        surrogate_framework (Pipeline): 
            Fitted suggorate framework to infer the quality of meta-points.
        
        acquisition_func (Callable[[np.ndarray, np.ndarray], np.ndarray]): 
            Acquisiton function to evaluate the promisingness of meta-points.
            It must be a function accepting the predicted scores and uncertainty,
            and returning a single array of promissingness scores.
    '''
    def __init__(
        self,
        surrogate_framework: Pipeline,
        acquisition_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ):
        self.surrogate_framework=surrogate_framework
        self.acquisition_func=acquisition_func


    def fit(self, metadata: pd.DataFrame, candidate_points: list[dict]) -> "MetadataEvaluator":
        '''
        Fit the instance with the candidate points and metadata.
        It's important the `candidate_points` and `metadata` order is respected,
        meaning the first candidate point form the first row of metadata,
        and so on. This is assumed true and not checked.

        Parameters:
            metadata (pd.DataFrame):
                Dataframe of hps (candidate points) + metafeatures

            candidate_points (list[dict]):
                List of points evaluated by the instance

        Returns:
            MetadataEvaluator: The fitted instance.
        '''
        self.metadata=metadata
        self.candidate_points=candidate_points
        self.n_candidate_points=len(candidate_points)
        self.is_fitted_=True
        return self


    def evaluate_candidates(self) -> np.ndarray:
        '''
        Evaluate the promisingness of the drawn candidate points,
        through the surrogate framework and acquisition function.
        This info is both returned and set internally.

        Returns:
            np.ndarray: The candidate points promisingness scores.
        '''
        check_is_fitted(self, "is_fitted_")
        pred_values, pred_uncertainty = self.surrogate_framework.predict(self.metadata)        
        self.promisingness_ = self.acquisition_func(pred_values, pred_uncertainty)
        return self.promisingness_


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
        check_is_fitted(self, "is_fitted_")
        self._check_promisingness_score_existence()

        if n_best > self.n_candidate_points:
            raise ValueError("'n_best' cannot be greater than the number of candidate points.")

        # argsort works in increasing order, so we reverse to have best --> worst direction
        top_idx = np.argsort(self.promisingness_, stable=True)[::-1][:n_best]
        return [self.candidate_points[idx] for idx in top_idx]
    

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
        check_is_fitted(self, "is_fitted_")
        self._check_promisingness_score_existence()

        if top > self.n_candidate_points:
            raise ValueError(f"'top' cannot be greater than the number of candidate points.")

        if n_proposed > top:
            raise ValueError(f"'n_proposed' cannot be greater than 'top'.")

        # argsort works in increasing order, so we reverse to have best --> worst direction
        top_idx = np.argsort(self.promisingness_, stable=True)[::-1][:top]
        top_points = [self.candidate_points[idx] for idx in top_idx]

        # we can skip random drawing in this scenario
        if top == n_proposed:
            return top_points
        
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(top, size=n_proposed, replace=False)
        return [top_points[idx] for idx in selected_idx]


    def propose_uniform_from_top(self, n_steps: int, step_size: int) -> list[dict[str, Any]]:
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
        check_is_fitted(self, "is_fitted_")
        self._check_promisingness_score_existence()

        working_range = n_steps * step_size
        
        if working_range > self.n_candidate_points:
            raise ValueError(
                "'n_steps * step_size' cannot be greater than the number of candidate points."
            )

        # argsort works in increasing order, so we reverse to have best --> worst direction
        ordered_idx = np.argsort(self.promisingness_, stable=True)[::-1]
        ordered_points = [self.candidate_points[idx] for idx in ordered_idx]
        return ordered_points[:working_range:step_size]


    def propose_random_uniform_from_top(
        self, 
        n_steps: int, 
        step_size: int,
        seed: int = 0
    ) -> list[dict[str, Any]]:
        '''
        Get the best `n_steps` points using a fixed `step_size` and a random seed.
        This propose strategy can be think of ordering the points 
        in the best --> worst direction, and then selecting a single item 
        randomly in the interval defined by the step_size, for n_steps times. 
        The utility therefore returns n_steps points.

        Parameters:
            n_steps (int): 
                Number of points returned by the utility.

            step_size (int):
                Size of the step defining the intervals.
            
            seed (int, optional):
                Seed used to randomly select the points inside the intervals.

        Returns:
            list[dict[str,Any]]: The list of the best points.
        '''
        check_is_fitted(self, "is_fitted_")
        self._check_promisingness_score_existence()

        working_range = n_steps * step_size
        
        if working_range > self.n_candidate_points:
            raise ValueError(
                "'n_steps * step_size' cannot be greater than the number of candidate points."
            )
        
        # argsort works in increasing order, so we reverse to have best --> worst direction
        ordered_idx = np.argsort(self.promisingness_, stable=True)[::-1]
        ordered_points = [self.candidate_points[idx] for idx in ordered_idx]
        
        rng = np.random.default_rng(seed)
        selected_points = []

        for i in range(n_steps):
            left_idx = i * step_size
            right_idx = left_idx + step_size
            selected_points.append(ordered_points[rng.integers(left_idx, right_idx)])

        return selected_points

    
    def _check_promisingness_score_existence(self) -> None:
        if not hasattr(self, "promisingness_"):
            raise ValueError(
                "The candidate points promisingness score has not been evaluated yet."
            )