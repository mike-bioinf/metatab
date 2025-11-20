from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from estimators.utils.types import Classifier
    from sklearn.pipeline import Pipeline
    from metatab_utils.types import XType



class EnsembleEstimatorMixin:
    '''
    Mixin for the ensemble estimators
    '''
    if TYPE_CHECKING:
        estimator_ : None ## WILL BE THE ENSEMBLE CLASS


    def predict_proba(self, X: XType) -> np.ndarray:
        pass
    
    def get_feature_names_in_(self) -> np.ndarray | None:
        pass

    def collect_sklearn_fit_info(self) -> dict:
        pass

    def collect_fit_preprocessing_info(self) -> dict:
        pass