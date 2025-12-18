from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab.estimators.utils.general import collect_sklearn_classification_fit_info
from metatab.preprocessing.collect import collect_fit_preprocessing_info

if TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline
    from metatab.metatab_utils.types import XType
    from metatab.estimators.utils.types import Classifier



class DefaultEstimatorMixin:
    '''
    Mixin class for the default estimator.

    Requirements:
    - Concrete class must define `estimator_` attribute (Classifier or Pipeline instance).
    - Concrete class MUST inherit from both TunedEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_ : Classifier | Pipeline


    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    
    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


    def get_feature_names_in_(self) -> np.ndarray | None:
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, "feature_names_in_", None)
    

    def collect_sklearn_fit_info(self) -> dict:
        '''
        Returns the `classes_`, `n_features_in_` and when existent 
        the `feature_names_in_` info in a dict with the keys names
        equal to the attributes names.
        '''
        check_is_fitted(self, "estimator_")
        return collect_sklearn_classification_fit_info(self.estimator_)


    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return collect_fit_preprocessing_info(self.estimator_, self.preprocessing)