from __future__ import annotations

from typing import Literal, TYPE_CHECKING, override
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from estimators.estimators.abstract_estimator import AbstractBaseEstimator

from estimators.estimators.utils import (
    add_string_to_params, 
    remove_string_from_params
)

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



class MyRandomForestClassifier(AbstractBaseEstimator):
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
        fixed_params = super().add_seed_to_fixed_params(copy=True)
        self.estimator_ = self._create_estimator(fixed_params)
        self.estimator_.fit(X, y)
        return self

    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return _create_rf_preprocessing_pipeline(self.preprocessing, fixed_params)
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_

       


class MyRandomizedRandomForestClassifier(AbstractBaseEstimator):
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
 
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyRandomizedRandomForestClassifier":
        fixed_params = super().add_seed_to_fixed_params(copy=True)
        self.estimator_ = RandomizedSearchCV(
            estimator=self._create_estimator(fixed_params),
            param_distributions=add_string_to_params(self.params_distributions, "randomforestclassifier__"),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            random_state=self.seed,
            **SKLEARN_RANDOM_SEARCH_FIXED_PARAMS
        )
        self.estimator_.fit(X, y)
        return self
        
    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return _create_rf_preprocessing_pipeline(self.preprocessing, fixed_params)

    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_.best_estimator_
    
    @override
    def get_best_hps(self) -> dict:
        check_is_fitted(self, "estimator_")
        return remove_string_from_params(self.estimator_.best_params_, "randomforestclassifier__")
        



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