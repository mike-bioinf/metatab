from typing import Literal, Union, TypeAlias

from metatab.metalearning.utils import  (
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams
)



MetaStrategy = Literal[
    "best", 
    "random_from_best", 
    "uniform_from_best"
]


MetaStrategyParams: TypeAlias = Union[
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams
]