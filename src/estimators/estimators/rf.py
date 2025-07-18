from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from estimators.estimators.abstract_estimator import AbstractEstimator
from estimators.estimators.utils import add_string_to_params

from estimators.estimators.utils import (
    create_density_filter_default_pipeline, 
    create_pca_default_pipeline
)

from estimators.estimators.params import (
    RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS,
    RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS,
    SKLEARN_RANDOM_SEARCH_FIXED_PARAMS
)

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from pathlib import Path



class MyRandomForestClassifier(AbstractEstimator):
    '''
    Class that wraps the random forest classifier.

    Attributes
    -------------
    estimator_ (Pipeline): 
        Fitted pipeline with RandomForestClassifier as head.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions = None,
        fixed_params: dict = RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomForestClassifier":
        self.estimator_ = self._create_estimator()
        self.estimator_.fit(X, y)
        return self

    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        return super()._classic_predict_proba(X)

    def save(self, filepath: str | Path) -> None:
        super().save(filepath)

    def _create_estimator(self) -> Pipeline:
        return _create_rf_preprocessing_pipeline(self.preprocessing, self.fixed_params)
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_

        


class MyRandomizedRandomForestClassifier(AbstractEstimator):
    '''
    Class that implements a random search over the random forest.

    Attributes
    ------------
    estimator_ (RandomizedSearchCV): 
        Fitted RandomizedSearchCV instance with a Pipeline as estimator
        which has a RandomForestClassifier as head.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions = RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)
 
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomizedRandomForestClassifier":
        self.estimator_ = RandomizedSearchCV(
            estimator=self._create_estimator(),
            param_distributions=add_string_to_params(self.params_distributions, "randomforestclassifier__"),
            cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=self.seed),
            **SKLEARN_RANDOM_SEARCH_FIXED_PARAMS
        )
        self.estimator_.fit(X, y)
        return self

    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        return super()._classic_predict_proba(X)

    def save(self, filepath: str | Path) -> None:
        super().save(filepath)
        
    def _create_estimator(self) -> Pipeline:
        return _create_rf_preprocessing_pipeline(self.preprocessing, self.fixed_params)

    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_.best_estimator_




def _create_rf_preprocessing_pipeline(
    preprocessing: Literal["base", "density_filter", "pca"],
    rf_params: dict
) -> Pipeline:
    '''Creates and returns the preprocessing pipelines with randomforest as head'''
    if preprocessing == "base":
        return make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            RandomForestClassifier(**rf_params)
        )
    elif preprocessing == "pca":
        return create_pca_default_pipeline(
            RandomForestClassifier, 
            rf_params
        )
    elif preprocessing == "density_filter":
        return create_density_filter_default_pipeline(
            "oversample",
            RandomForestClassifier,
            rf_params
        )
    else:
        raise ValueError("Unsupported preprocessing.")