from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from pathlib import Path
    from sklearn.pipeline import Pipeline
    from hp_search.types import MetaAlgo, MetaStrategy, MetaStrategyParams



@dataclass
class TuneConfiguration:
    '''
    Dataclass that manages some of the SearchCV input parameters.
    In particular hosts the parameters that are managed by the
    AbstractBaseEStimator at init level.  
    '''
    algo: MetaAlgo
    n_iter: int
    n_cv_repeats: int
    n_cv_folds: int
    params_distributions: dict
    meta_strategy: MetaStrategy
    meta_strategy_params: None | MetaStrategyParams = None
    meta_surrogate_model: None | Pipeline = None
    meta_seed: int = 42
    raise_error_during_search: None | bool = None
    build_df_search: None | bool = None
    refit_with_best_hps: None | bool = None
    save_realtime_df_search_filepath: None | str | Path = None



@dataclass
class EarlyStopConfiguration:
    '''Dataclass that manages the early stop configuration'''
    early_stop_rounds: int = 100
    validation_set_size: float = 0.3



@dataclass
class EnsembleConfiguration:
    '''Dataclass that manages the ensemble configuration'''
    pass