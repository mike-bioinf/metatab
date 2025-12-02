from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from metalearning.constants import METASTRATEGIES

if TYPE_CHECKING:
    from metalearning.types import MetaStrategy, MetaStrategyParams



@dataclass
class BestMetaStrategyParams:
    '''
    Dataclass to use to pass custom parameters for the 'best' meta-optimization strategy.
    n_candidate_points (int): Number of points to draw as candidates.
    '''
    n_candidate_points: int


@dataclass
class RandomFromBestMetaStrategyParams:
    '''
    Dataclass to use to pass custom parameters for the 'random_from_best' meta-optimization strategy.
    n_candidate_points (int): Number of points to draw as candidates.
    top (int): Number of top points from which draw the random ones.
    seed (int): Control the randomness of the point selection procedure.   
    '''
    n_candidate_points: int
    top: int
    seed: int


@dataclass
class UniformFromBestMetaStrategyParams:
    '''
    Dataclass to use to pass custom parameters for the 'uniform_from_best' meta-optimization strategy.
    n_candidate_points (int): Number of points to draw as candidates.
    step_size (int): Size of the step used to choose points starting from the best.
    '''
    n_candidate_points: int
    step_size: int



def check_meta_strategy(meta_strategy: str) -> None:
    if meta_strategy not in METASTRATEGIES:
        raise ValueError(f"meta_strategy must be one of: {METASTRATEGIES}")



def check_meta_strategy_params(
    meta_strategy: MetaStrategy, 
    meta_strategy_params: None | MetaStrategyParams, 
    safe_none_params: bool
) -> None:
    '''
    Check that `meta_strategy` and `meta_strategy_params` are compatible.
    If `safe_none_params` is True then `meta_strategy_params` can be None.
    '''
    if meta_strategy_params is None and safe_none_params:
        return None
    
    if (
        meta_strategy == "best" and 
        not isinstance(meta_strategy_params, BestMetaStrategyParams)
    ):
        raise ValueError((
            "With 'best' meta_strategy a 'BestMetaStrategyParams'"
            " object is expected in meta_strategy_params."
        ))
    
    elif (
        meta_strategy == "random_from_best" and 
        not isinstance(meta_strategy_params, RandomFromBestMetaStrategyParams)
    ):
        raise ValueError((
            "With 'random_from_best' meta_strategy a 'RandomFromBestMetaStrategyParams'"
            " object is expected in meta_strategy_params."
        ))
    
    elif (
        meta_strategy == "uniform_from_best" and
        not isinstance(meta_strategy_params, UniformFromBestMetaStrategyParams)
    ):
        raise ValueError((
            "With 'uniform_from_best' meta_strategy a 'UniformFromBestMetaStrategyParams'"
            " object is expected in meta_strategy_params."
        ))