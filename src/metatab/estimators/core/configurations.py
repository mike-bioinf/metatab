from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass

if TYPE_CHECKING:
    from pathlib import Path
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams



@dataclass
class TuneConfiguration:
    '''
    Dataclass that manages the tune configuration.
    Given in input to `AbstractBaseEstimator`.
    '''
    algo: Literal["random", "tpe", "meta"]
    n_iter: int
    n_cv_repeats: int
    n_cv_folds: int
    params_distributions: dict
    meta_strategy: MetaStrategy = "best"
    meta_strategy_params: None | MetaStrategyParams = None
    meta_surrogate_model: None | str | Path = None
    meta_seed: int = 42
    raise_error_during_search: None | bool = None
    build_df_search: None | bool = None
    refit_with_best_hps: None | bool = None



@dataclass
class EarlyStopConfiguration:
    '''
    Dataclass that manages the early stop configuration.
    Given in input to `AbstractBaseEstimator`.
    '''
    early_stop_rounds: int = 100
    validation_set_size: float = 0.3



@dataclass
class EnsembleConfiguration:
    '''
    Dataclass that manages the ensemble configuration.
    Given in input to `AbstractBaseEstimator`.
    '''
    name: str
    algo: Literal["random", "meta"]
    n_members: int
    save_path: str | Path
    params_distributions: dict
    meta_strategy: MetaStrategy = "random_uniform_from_best"
    meta_strategy_params: None | MetaStrategyParams = None
    meta_surrogate_model: None | str | Path = None
    meta_seed: int = 42
    meta_features: None | dict = None
    meta_candidate_points: None | list[dict] = None
    time_limit: int = 10_000_000
    log: int = 20
    raise_error_fit_member: bool = False
    raise_error_void_ensemble: bool = True