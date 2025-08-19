from __future__ import annotations

from typing import Literal, TYPE_CHECKING, override
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from estimators.abstract_estimator import AbstractBaseEstimator
from estimators.searchcv import SearchCV
from estimators.params import TuningParams, DefaultParams

from estimators.utils import (
    create_density_filter_default_pipeline, 
    create_pca_default_pipeline
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
        n_threads: int,
        tune_configuration = None,
        fixed_params: dict = DefaultParams.RANDOM_FOREST_DEFAULT_PARAMS
    ):
        super().__init__(preprocessing, seed, n_threads, tune_configuration, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomForestClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)
        self.estimator_ = _create_rf_pipeline(self.preprocessing, fixed_params)
        self.estimator_.fit(X, y)
        return self
       


class MyTunedRandomForestClassifier(AbstractBaseEstimator):
    '''
    Class that implements random forest with HPO.

    Attributes
    -----------------
    estimator_ (SeachCV): Fitted SearchCV instance
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_threads: int,
        tune_configuration: dict,
        fixed_params: dict = TuningParams.RANDOM_FOREST_FIXED_PARAMS  
    ):
        super().__init__(preprocessing, seed, n_threads, tune_configuration, fixed_params)
 
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTunedRandomForestClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)

        self.estimator_ = SearchCV(
            clf_or_pipe=_create_rf_pipeline(self.preprocessing, fixed_params),
            algo=self.tune_configuration["algo"],
            params_distributions=self.tune_configuration["params_distributions"],
            random_state_parameter="random_state",
            n_iter=self.tune_configuration["n_iter"],
            n_cv_repeats=self.tune_configuration["n_repeats"],
            n_cv_splits=self.tune_configuration["n_splits"],
            seed=self.seed,
            metric_to_minimize="logloss",
            early_stop_on_validation_set=False
        )

        self.estimator_.fit(X, y)
        return self
    
    @override
    def get_best_hps(self) -> dict:
        check_is_fitted(self, "estimator_")
        return self.estimator_.best_params_
    


def _create_rf_pipeline(
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