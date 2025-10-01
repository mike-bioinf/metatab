from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.ensemble import RandomForestClassifier
from estimators.abstract_estimator import AbstractBaseEstimator
from hp_search.searchcv import SearchCV
from estimators.params import TuningParams, DefaultParams
from estimators.utils import create_default_pipeline

if TYPE_CHECKING:
    import pandas as pd




class MyRandomForestClassifier(AbstractBaseEstimator):
    '''
    Class that wraps the random forest classifier.

    Attributes
    -------------
    estimator_ (Pipeline): Fitted pipeline with RandomForestClassifier as head.
    '''
    fixed_params = DefaultParams.RANDOM_FOREST_DEFAULT_PARAMS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyRandomForestClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)
        self.estimator_ = create_default_pipeline(
            preprocessing=self.preprocessing, 
            density_feature_selector_strategy="oversample", 
            classifier=RandomForestClassifier, 
            classifier_params=fixed_params
        )
        self.estimator_.fit(X, y)
        return self
       


class MyTunedRandomForestClassifier(AbstractBaseEstimator):
    '''
    Class that implements random forest with HPO.

    Attributes
    -----------------
    estimator_ (SeachCV): Fitted SearchCV instance.
    '''
    fixed_params = TuningParams.RANDOM_FOREST_FIXED_PARAMS 
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTunedRandomForestClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)
        self.estimator_ = SearchCV(
            clf_or_pipe=create_default_pipeline(
                preprocessing=self.preprocessing, 
                density_feature_selector_strategy="oversample", 
                classifier=RandomForestClassifier, 
                classifier_params=fixed_params
            ),
            type_clf_or_pipe_preprocessing=self.preprocessing,
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