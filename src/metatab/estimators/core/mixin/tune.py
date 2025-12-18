from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab.estimators.utils.general import collect_sklearn_classification_fit_info
from metatab.preprocessing.collect import collect_fit_preprocessing_info

if TYPE_CHECKING:
    from metatab.hp_search.searchcv import SearchCV
    from metatab.metatab_utils.types import XType



class TunedEstimatorMixin:
    '''
    Mixin class for the tuned estimator.
    
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
        return collect_fit_preprocessing_info(
            self.estimator_.best_estimator_,
            self.preprocessing,
        )

    
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
        return collect_sklearn_classification_fit_info(self.estimator_.best_estimator_)


    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return self.estimator_.best_estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        self._check_estimator_is_refitted()
        return self.estimator_.best_estimator_.predict_proba(X)
    

    def _check_estimator_is_refitted(self) -> None:
        if not self.estimator_.refit_with_best_hps:
            raise ValueError("SearchCv instance has the refitting option disabled.")        