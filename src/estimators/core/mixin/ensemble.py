from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from estimators.utils.general import collect_sklearn_classification_fit_info
from preprocessing.collect import collect_fit_preprocessing_info

if TYPE_CHECKING:
    import numpy as np
    from metatab_utils.types import XType
    from ensemble.single import EnsembleEstimator



class EnsembleEstimatorMixin:
    '''
    Mixin for the ensemble estimators.

    Requirements:
    - Concrete class must define `estimator_` attribute (Classifier or Pipeline instance).
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


    def collect_ensemble_fit_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        return {
            "is_void": self.estimator_.is_void_,
            "fit_time": self.estimator_.fit_time_,
            "successful_members": self.estimator_.successful_members_, 
            "failed_members": self.estimator_.failed_members_,
            "successful_hps_confs": self.estimator_.successful_hps_confs_,
            "failed_hps_confs": self.estimator_.failed_hps_confs_,
            "df_members": self.estimator_.df_members_
        }

    
    def collect_fit_preprocessing_info(self) -> dict:
        # the check is useful also in this case
        check_is_fitted(self, "estimator_")
        self.estimator_._check_on_predict_calls()
        return collect_fit_preprocessing_info(
            clf_or_pipe=self.estimator_._save_path / self.estimator_.successful_members_[0], 
            preprocessing=self.preprocessing
        )