from __future__ import annotations

import warnings
import json
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from preprocessing import DensityFeatureSelector
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from estimators.abstract_estimator import AbstractBaseEstimator
from estimators.params import DefaultParams, TuningParams
from hp_search.searchcv import SearchCV

from finetabpfn import AesFineTunedTabPFNClassifier

from estimators.utils import (
    create_pca_default_pipeline, 
    create_density_filter_default_pipeline
)

if TYPE_CHECKING:
    import numpy as np



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
    tabpfn_params: dict,
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"]
) -> TabPFNClassifier | Pipeline:
    if preprocessing == "base":
        return TabPFNClassifier(**tabpfn_params)
    elif preprocessing == "pca":
        return create_pca_default_pipeline(TabPFNClassifier, tabpfn_params)
    elif preprocessing == "density_filter":
        return create_density_filter_default_pipeline(
            density_feature_selector_strategy, 
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
        self.estimator_ = create_tabpfn_estimator(self.preprocessing, fixed_params, "oversample")
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
            clf_or_pipe=create_tabpfn_estimator(self.preprocessing, fixed_params, "undersample"),  # undersample to speed up
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
    
    @staticmethod
    def _create_autotabpfn_estimator(
        preprocessing: Literal["base", "density_filter", "pca"], 
        params: dict
    ) -> AutoTabPFNClassifier | Pipeline:
        if preprocessing == "base":
            return AutoTabPFNClassifier(**params)
        elif preprocessing == "pca":
            raise ValueError("PCA preprocessing is not possible with AutoTabPFNClassifier.")
        elif preprocessing == "density_filter":
            return create_density_filter_default_pipeline(
                "undersample", # to speed up 
                AutoTabPFNClassifier, 
                params
            )
        else:
            raise ValueError("Unsupported preprocessing.")
     



class SingleDatasetAesFineTunedTabpfnClassifier(ClassifierMixin, BaseEstimator):
    '''
    Sklearn like-class that wraps the AesFineTunedTabPFNClassifier class, 
    working exclusevely on a single dataset and using the finetune data 
    as context during inference.

    Parameters
    ------------------
    finetuned_classifier (AesFineTunedTabPFNClassifier):
        Instance of the AesFineTunedTabPFNClassifier class.

    Attributes
    -------------------
    _X_train (pd.DataFrame | np.ndarray): X finetune data.
    _y_train (pd.Series): y finetune data.
    n_features_in_ (int): Number of features of the X finetune data.
    feature_names_in_ (list[str]): Names of the features of the X finetuned data if a pandas DataFrame.
    n_classes_ (int): Number of classes of the y finetune data.
    is_fitted_ (bool): Whether the instance is fitted.
    finetuned_classifier_ (AesFineTunedTabPFNClassifier): Fitted finetuned_classifier instance.
    stats_finetune_ (dict[str, Any]): Dict with the finetune stastistics.
    df_finetune_ (pd.DataFrame): Dataframe with the training finetune statistics.
    '''
    def __init__(self, finetuned_classifier: AesFineTunedTabPFNClassifier):
        self.finetuned_classifier=finetuned_classifier

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series) -> "SingleDatasetAesFineTunedTabpfnClassifier":
        self.finetuned_classifier_ = deepcopy(self.finetuned_classifier)
        self.finetuned_classifier_.fit(X, y, use_for_validation=True)
        self._X_train = X
        self._y_train = y
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.index.to_list()
        self.n_classes_ = y.unique().size
        self.is_fitted_ = True
        self.stats_finetune_ = self.finetuned_classifier_.stats_finetune_
        self.df_finetune_ = self.stats_finetune_["df_finetune"]
        # we store the df_finetune separately
        del self.stats_finetune_["df_finetune"]
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        return self.finetuned_classifier_.predict_proba(
            X, 
            X_contest=self._X_train, 
            y_contest=self._y_train, 
            categorical_features_indices=[]
        )
        


class MyAesFineTunedTabPFNClassifier(AbstractBaseEstimator):
    '''
    Finetuned tabpfn classifier with an adaptive early stopping strategy on a validation set.
    
    Attributes
    ----------------
    estimator_ (Pipeline): Fitted Pipeline instance.
    '''
    fixed_params = DefaultParams.AESFINETUNEDTABPFN_DEFAULT_PARAMS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyAesFineTunedTabPFNClassifier":
        self.estimator_ = self._create_estimator(
            preprocessing=self.preprocessing, 
            params=self._update_fixed_params_with_seed_and_nthreads()
        )
        self.estimator_.fit(X, y)
        return self
    

    def _update_fixed_params_with_seed_and_nthreads(self) -> dict:
        params = deepcopy(self.fixed_params)
        params["tabpfn_classifier_params"]["random_state"] = self.seed
        params["tabpfn_classifier_params"]["n_jobs"] = self.n_threads
        params["seed"] = self.seed
        return params


    def save_finetune_stats(self, txt_filepath: str | Path, json_filepath: str | Path) -> None:
        check_is_fitted(self, "estimator_")
        finetuned_instance = self.estimator_[-1]
        finetuned_instance.df_finetune_.to_csv(txt_filepath, sep="\t", index=False)
        with open(json_filepath, "w") as f:
            json.dump(finetuned_instance.stats_finetune_, f, indent=4)


    @staticmethod
    def _create_estimator(
        preprocessing: Literal["base", "density_filter", "pca"],
        params: dict
    ) -> Pipeline:
        if preprocessing == "base":
            return make_pipeline(
                VarianceThreshold(),
                SingleDatasetAesFineTunedTabpfnClassifier(
                    AesFineTunedTabPFNClassifier(**params)
                )
            )
        
        elif preprocessing == "pca":
            raise ValueError(
                "PCA preprocessing is not possible with AesFineTunedTabPFNClassifier."
            )
        
        elif preprocessing == "density_filter":
            return make_pipeline(
                VarianceThreshold(), 
                DensityFeatureSelector(
                    n_target_cols=500, 
                    strategy="undersample", # to speed up
                    error_on_empty=True
                ),
                SingleDatasetAesFineTunedTabpfnClassifier(
                    AesFineTunedTabPFNClassifier(**params)
                )
            )
        
        else:
            raise ValueError("Unsupported preprocessing.")