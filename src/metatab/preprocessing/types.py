from typing import Literal


PreprocessingStrategy = Literal[
    "estimator_default",
    "base",
    "density_filter",
    "pca",
    "no"
]

ResolvedPreprocessingStrategy = Literal[
    "base",
    "density_filter",
    "pca",
    "no"
]