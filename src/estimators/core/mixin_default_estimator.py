from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from estimators.utils.types import Classifier
    from sklearn.pipeline import Pipeline



class DefaultEstimatorMixin:
    '''
    Mixin class for the default estimators

    Requirements:
    - Concrete class must define `estimator_` attribute (Classifier or Pipeline instance).
    - Concrete class MUST inherit from both TunedEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_ : Classifier | Pipeline


    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)
    

    def get_feature_names_in_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_names_in_


    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return super().collect_fit_preprocessing_info(self.estimator_)