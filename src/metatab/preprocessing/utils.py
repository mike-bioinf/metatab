from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metatab.preprocessing.types import PreprocessingStrategy, ResolvedPreprocessingStrategy
    from metatab.estimators.utils.types import EstimatorType



def get_estimator_default_preprocessing(type_estimator: EstimatorType):
    if type_estimator == "tabpfn":
        return "density_filter"
    else:
        return "base"


def resolve_preprocessing_info(preprocessing: PreprocessingStrategy) -> ResolvedPreprocessingStrategy:
    '''Resolves and returns the explicit preprocessing info'''
    return get_estimator_default_preprocessing(preprocessing) \
        if preprocessing == "estimator_default" \
        else preprocessing