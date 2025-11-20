from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline
    from metatab_utils.types import XType
    from estimators.utils.types import Classifier



class DefaultEstimatorMixin:
    '''
    Mixin class for the default estimators

    Requirements:
    - Concrete class must define `estimator_` attribute (Classifier or Pipeline instance).
    - Concrete class MUST inherit from both TunedEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_ : Classifier | Pipeline


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
        out = {
            "classes_": self.estimator_.classes_,
            "n_features_in_": self.estimator_.n_features_in_,
        }
        feature_names_in = getattr(self.estimator_, "feature_names_in_", None)
        if feature_names_in is not None:
            out["feature_names_in_"] = feature_names_in
        return out


    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return super().collect_fit_preprocessing_info(self.estimator_)