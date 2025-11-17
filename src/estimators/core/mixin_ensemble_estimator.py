from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from estimators.utils.types import Classifier
    from sklearn.pipeline import Pipeline



class EnsembleEstimatorMixin:
    '''
    Mixin for the ensemble estimators
    '''
    if TYPE_CHECKING:
        estimator_ : None ## WILL BE THE ENSEMBLE CLASS

    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        pass
    

    def get_feature_names_in_(self) -> np.ndarray:
        pass


    def collect_fit_preprocessing_info(self) -> dict:
        pass