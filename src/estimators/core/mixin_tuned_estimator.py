from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from hp_search.searchcv import SearchCV
    from metatab_utils.types import XType



class TunedEstimatorMixin:
    '''
    Mixin providing methods for tuned estimators.
    
    Requirements:
    - Concrete class must define `estimator_` attribute (SearchCV instance).
    - Concrete class MUST inherit from both TunedEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_: SearchCV

    
    def get_best_hps(self) -> dict:
        check_is_fitted(self, "estimator_")
        return self.estimator_.best_params_
    
    
    def get_search_losses(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return np.array(self.estimator_.search_losses_)
    
    
    def get_refit_time(self) -> float:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return self.estimator_.refit_time_
    

    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return super().collect_fit_preprocessing_info(self.estimator_.best_estimator_)

    
    def get_feature_names_in_(self) -> np.ndarray | None:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return getattr(self.estimator_.best_estimator_, "feature_names_in_", None)
    

    def collect_sklearn_fit_info(self) -> dict:
        '''
        Returns the `classes_`, `n_features_in_` and when existent 
        the `feature_names_in_` info in a dict with the keys names
        equal to the attributes names.
        '''
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        out = {
            "classes_": self.estimator_.best_estimator_.classes_,
            "n_features_in_": self.estimator_.best_estimator_.n_features_in_,
        }
        feature_names_in = getattr(self.estimator_.best_estimator_, "feature_names_in_", None)
        if feature_names_in is not None:
            out["feature_names_in_"] = feature_names_in
        return out


    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return self.estimator_.best_estimator_.predict_proba(X)


    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return self.estimator_.best_estimator_.predict(X)


    def _check_estimator_is_refitted(self) -> None:
        if not self.estimator_.refit_with_best_hps:
            raise ValueError("SearchCv instance has the refitting option disabled.")        