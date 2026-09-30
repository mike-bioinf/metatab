from __future__ import annotations

import time
import numpy as np
from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
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


    def _predict_proba_best_at_k(self, X: XType) -> list[dict]:
        '''Predict with the best estimators at k'''
        check_is_fitted(self, "estimator_")

        if not self.estimator_.refit_at_k:
            raise ValueError("No refit at k indication is present in SearchCV instance")
        
        res = []
        for dict_at_k in self.estimator_.best_at_k_:
            classifier_at_k: Pipeline = dict_at_k["estimator"]
            t = time.time()
            preds_at_k = classifier_at_k.predict_proba(X)
            predict_time = time.time() - t
            res.append({
                "k": dict_at_k["k"],
                "pred_proba": preds_at_k,
                "fit_time": dict_at_k["fit_time"],
                "predict_time": predict_time,
                "inner_val_loss": dict_at_k["loss"]
            })

        return res

               