from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab.preprocessing.collect import collect_fit_preprocessing_info

if TYPE_CHECKING:
    import numpy as np
    from metatab.metatab_utils.types import XType
    from metatab.ensemble.single import EnsembleEstimator



class EnsembleEstimatorMixin:
    '''
    Mixin for the ensemble estimators.

    Requirements:
    - Concrete class must define `estimator_` attribute (EnsembleEstimator instance).
    - Concrete class MUST inherit from both EnsembleEstimatorMixin AND AbstractBaseEstimator.
    '''
    if TYPE_CHECKING:
        estimator_ : EnsembleEstimator

    
    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)
    

    def get_members_predicted_probabilities(self, X: XType) -> dict[str, np.ndarray]:
        check_is_fitted(self, "estimator_")
        return self.estimator_.get_members_predicted_probabilities(X)


    def collect_ensemble_fit_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return {
            "is_void_": self.estimator_.is_void_,
            "is_cleaned_": self.estimator_.is_cleaned_,
            "fit_time_": self.estimator_.fit_time_,
            "successful_members_": self.estimator_.successful_members_, 
            "failed_members_": self.estimator_.failed_members_,
            "successful_hps_confs_": self.estimator_.successful_hps_confs_,
            "failed_hps_confs_": self.estimator_.failed_hps_confs_,
            "df_members_": self.estimator_.df_members_
        }

    
    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        # this check is useful also in this case
        self.estimator_._check_on_predict_calls()
        model = self.estimator_._save_path / f"{self.estimator_.successful_members_[0]}.pkl"
        return collect_fit_preprocessing_info(
            self.estimator_._try_load_model(model), 
            self.preprocessing
        )