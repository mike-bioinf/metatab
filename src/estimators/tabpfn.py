from __future__ import annotations

import warnings
from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from estimators.abstract_estimator import AbstractBaseEstimator
from estimators.params import DefaultParams, TuningParams
from hp_search.searchcv import SearchCV

from estimators.utils import (
    create_pca_default_pipeline, 
    create_density_filter_default_pipeline
)

if TYPE_CHECKING:
    import pandas as pd



def suppress_sklearn_and_tabpfn_warnings(func):
    '''
    Decorator to filter sklearn future deprecation warnings,
    and tabpfn loading and ignore training limits warning.
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



def create_tabpfn_estimator(
    preprocessing: Literal["base", "density_filter", "pca"], 
    tabpfn_params: dict 
) -> TabPFNClassifier | Pipeline:
    if preprocessing == "base":
        return TabPFNClassifier(**tabpfn_params)
    elif preprocessing == "pca":
        return create_pca_default_pipeline(TabPFNClassifier, tabpfn_params)
    elif preprocessing == "density_filter":
        return create_density_filter_default_pipeline(
            "oversample", 
            TabPFNClassifier, 
            tabpfn_params
        )
    else:
        raise ValueError("Unsupported preprocessing.")



class MyTabPFNClassifier(AbstractBaseEstimator):
    '''
    Class that wraps the base TabPFNClassifier.

    Attributes
    -----------
    estimator_ (TabPFNClassifier | Pipeline):
        Fitted estimator. Is a TabPFNClassifier instance in case
        of "base" preprocessing or a Pipeline instance otherwise.  
    '''
    fixed_params = DefaultParams.TABPFN_DEFAULT_PARAMS
 
    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTabPFNClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)
        self.estimator_ = create_tabpfn_estimator(self.preprocessing, fixed_params)
        self.estimator_.fit(X, y)
        return self
        


class MyTunedTabPFNClassifier(AbstractBaseEstimator):
    '''
    TabPFNClassifier with HPs tuning.

    Attributes
    ----------------
    estimator_ (SearchCV): Fitted SearchCV instance.
    '''
    fixed_params = TuningParams.TABPFN_FIXED_PARAMS

    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTunedTabPFNClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_threads=True, copy=True)
        self.estimator_ = SearchCV(
            clf_or_pipe=create_tabpfn_estimator(self.preprocessing, fixed_params),
            algo=self.tune_configuration["algo"],
            params_distributions=self.tune_configuration["params_distributions"],
            n_iter=self.tune_configuration["n_iter"],
            n_cv_repeats=self.tune_configuration["n_repeats"],
            n_cv_splits=self.tune_configuration["n_splits"],
            random_state_parameter="random_state",
            seed=self.seed,
            metric_to_minimize="logloss",
            early_stop_on_validation_set=False
        )
        self.estimator_.fit(X, y)
        return self



class MyAutoTabPFNClassifier(AbstractBaseEstimator):
    '''
    Autogluon ensemble (stacking + Caruana selection) of tabpfn classifiers.
    
    Attributes
    ----------------
    estimator_ (AutoTabPFNClassifier|Pipeline): Fitted AutoTabPFNClassifier or Pipeline instance.
    '''
    fixed_params = DefaultParams.AUTOTABPFN_DEFAULT_PARAMS

    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyAutoTabPFNClassifier":
        fixed_params = self.update_fixed_params(up_seed=True, copy=True)
        self.estimator_ = self._create_autotabpfn_estimator(self.preprocessing, fixed_params)
        # to suppress automatic categorical features inferring
        fit_args = {"autotabpfnclassifier__categorical_feature_indices": []}\
            if isinstance(self.estimator_, Pipeline)\
            else {"categorical_feature_indices": []}
        self.estimator_.fit(X, y, **fit_args)
        return self
    
    def _create_autotabpfn_estimator(
        preprocessing:  Literal["base", "density_filter", "pca"], 
        params: dict
    ) -> AutoTabPFNClassifier | Pipeline:
        if preprocessing == "base":
            return AutoTabPFNClassifier(**params)
        elif preprocessing == "pca":
            raise ValueError("PCA preprocessing is not possible with AutoTabPFNClassifier")
        elif preprocessing == "density_filter":
            return create_density_filter_default_pipeline(
                "undersample", # to speed up 
                AutoTabPFNClassifier, 
                params
            )
        else:
            raise ValueError("Unsupported preprocessing.")