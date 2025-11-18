from typing import Literal, Union, TypeAlias

from hp_search.utils import  (
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams
)


MetaAlgo = Literal["random", "tpe", "meta"]

MetaStrategy = Literal["best", "random_from_best", "uniform_from_best"]

MetaStrategyParams: TypeAlias = Union[
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams
]