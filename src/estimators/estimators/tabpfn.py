from __future__ import annotations

import warnings
from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier
from estimators.estimators.abstract_estimator import AbstractEstimator
from estimators.estimators.params import TABPFN_CLASSIFIER_FIXED_PARAMS

from estimators.estimators.utils import (
    create_pca_default_pipeline, 
    create_density_filter_default_pipeline
)

if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
    import pandas as pd



def suppress_sklearn_and_tabpfn_warnings(func):
    '''
    Decorator to filter sklearn future deprecation warnings,
    and tabpfn loading and ignore limits warning.
    '''
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*", module=".*tabpfn.*loading")
            warnings.filterwarnings(
                action="ignore", 
                message=".*is greater than the maximum Number of features 500 supported by the model.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper



class MyTabPFNClassifier(AbstractEstimator):
    '''
    Class that wraps the base TabPFNClassifier.

    Attributes
    -----------
    estimator_ (TabPFNClassifier | Pipeline):
        Fitted estimator. Is a TabPFNClassifier instance in case
        of "base" preprocessing or a Pipeline instance otherwise.  
    '''
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"], 
        seed: int, 
        params_distributions = None, 
        fixed_params = TABPFN_CLASSIFIER_FIXED_PARAMS
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)


    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyTabPFNClassifier":
        self.estimator_ = self._create_estimator()
        self.estimator_.fit(X, y)
        return self


    def predict_proba(self, X, **kwargs) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X, **kwargs)


    def save(self, filepath: str | Path) -> None:
        super().save(filepath)


    def _create_estimator(self) -> TabPFNClassifier | Pipeline:
        if self.preprocessing == "base":
            return TabPFNClassifier(**self.fixed_params)
        elif self.preprocessing == "pca":
            return create_pca_default_pipeline(TabPFNClassifier, self.fixed_params)
        elif self.preprocessing == "density_filter":
            return create_density_filter_default_pipeline(
                "oversample", 
                TabPFNClassifier, 
                self.fixed_params
            )
        else:
            raise ValueError("Unsupported preprocessing.")
