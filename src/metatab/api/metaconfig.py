from dataclasses import dataclass
from pathlib import Path
from metatab.metalearning.utils import check_meta_strategy, check_meta_strategy_params
from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
from metatab.metalearning.utils import BestMetaStrategyParams, RandomFromBestMetaStrategyParams, UniformFromBestMetaStrategyParams, RandomUniformFromBestMetaStrategyParams


@dataclass
class MetaConfig:
    '''
    Meta-framework configuration class used in meta-tuning/ensembling applications.

    Parameters:
        strategy (MetaStrategy, optional):
            Set the strategy used by the meta-framework to select points.
            - "best": select the top-n configurations.
            - "random_from_best": random selection from the top.
            - "uniform_from_best": uniform step selection from the top.
            - "random_uniform_from_best": random selection within uniform intervals from the top.
            
        strategy_params (None | MetaStrategyParams, optional):
            Meta strategy specifics in form of dataclass.
            If None the default specifics are applied.

        surrogate_model (None | str | Path, optional):
            Path of the surrogate model used by the meta-framework.
            If str or Path, then the object pointed by the path is used as surrogate model.
            This must be a joblib serialized object.
            If None the "default" surrogate model is used.

        seed (int, optional):
            Seed used in the candidate points drawing process.
            The default value of 42 is the one used to generate our prior.
            Using this value allows to evaluate points tested beforehand. 
            Therefore is highly suggested to not modify this value.
            Note: If the parameter is left at its default but the number of 
            candidate points (set via `strategy_params`) differs from the 
            default of 1500, the following occurs.
            - If the number of candidate points is less than 1500, a subset of the prior points is selected.
            - If the number exceeds 1500, "new" points are drawn in addition to the prior points.
    '''
    n_candidate_points: int = 500 ##REFACTOR: avere questa info qui???
    strategy: MetaStrategy = "best"
    strategy_params: None | MetaStrategyParams = None
    surrogate_model: None | str | Path = None
    seed: int = 42

    @classmethod
    def build_from_dict(cls, dictionary: dict) -> "MetaConfig":
        return cls(**dictionary)
    
    def _check(self) -> None:
        check_meta_strategy(self.strategy)
        check_meta_strategy_params(self.strategy, self.strategy_params, safe_none_params=True)