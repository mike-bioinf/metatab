from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab.preprocessing.collect import collect_fit_preprocessing_info

if TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline
    from metatab.metatab_utils.types import XType



class DefaultEstimatorMixin:
    '''
    Mixin class for the default estimator.

    Requirements:
    - Concrete class must define the `estimator_` attribute (Pipeline instance).
    - Concrete class MUST inherit from both TunedEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_ : Pipeline


    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    
    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return collect_fit_preprocessing_info(self.estimator_, self.preprocessing)