from typing import Literal, Union, TypeAlias

from metatab.metalearning.utils import  (
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams,
    RandomUniformFromBestMetaStrategyParams
)


MetaStrategy = Literal[
    "best",
    "random_from_best",
    "uniform_from_best",
    "random_uniform_from_best"
]

MetaStrategyParams: TypeAlias = Union[
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams,
    RandomUniformFromBestMetaStrategyParams
]