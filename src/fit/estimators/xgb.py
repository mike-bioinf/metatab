import numpy as np
import pandas as pd
from typing import Literal
from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from fit.estimators.random_search import MyRandomSearchCV
from fit.preprocessing import DensityFeatureSelector
from fit.estimators.abstract_estimator import AbstractEstimator
from fit.estimators.utils_estimators import add_string_to_params

from fit.constants import (
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS, 
    ES_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS,
    RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS,
    SKLEARN_RANDOM_SEARCH_FIXED_PARAMS
)



class MyESRandomizedXGBClassifier(AbstractEstimator):
    '''
    Class that uses a custom implementation of random search cv (MyRandomSearchCV)
    with the XGBClassifier. This custom implementation allows and enforces the 
    use of early stopping on a validation set for the ensemble building process.

    Attributes
    ------------------    
    estimator_ (MyRandomSearchCV): Fitted MyRandomSearchCV instance. 
    '''
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions: dict = RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = ES_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)


    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyESRandomizedXGBClassifier":
        '''
        Fit the estimator on X and y.
        Here kwargs is not used but is present for compatibility with other estimators.
        Set the "estimator_" attribute.
        '''
        n_classes = y.unique().size

        # updating fixed params with the fit-specific params
        fixed_params = deepcopy(self.fixed_params)
        
        if n_classes == 2:
            # here we must NOT specify num_class otherwise strange behaviours
            fixed_params["objective"] = "binary:logistic"
        else:
            fixed_params["objective"] = "multi:softprob"
            fixed_params["num_class"] = n_classes
        
        # pass a seed here in splitter and NOT a RandomState
        splitter = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=self.seed)
        preprocessing_pipeline = self._create_preprocessing_pipeline()

        estimator = MyRandomSearchCV(
            classifier=XGBClassifier,
            fixed_params_classifier=fixed_params,
            param_distributions=self.params_distributions,
            preprocessing_pipeline=preprocessing_pipeline,
            splitter = splitter,
            scorer="logloss",
            n_iter=100,
            refit=True,
            seed=self.seed
        )

        self.estimator_ = estimator.fit(X, y)
        return self
    

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X, **kwargs)
        
        
    def _create_preprocessing_pipeline(self) -> Pipeline:
        '''
        Create the preprocessing pipeline based on the "preprocessing".
        Returns a Pipeline object.
        '''
        if self.preprocessing == "base":
            return make_pipeline(VarianceThreshold())
        elif self.preprocessing == "pca":
            return make_pipeline(
                VarianceThreshold(), 
                StandardScaler(),
                PCA(svd_solver="full", n_components=0.95)
            )
        elif self.preprocessing == "density_filter":
            return make_pipeline(
                VarianceThreshold(),
                DensityFeatureSelector(n_target_cols=500, strategy="oversample")
            )
        else:
            raise ValueError("Unsupported preprocessing.")
        



class MyRandomizedXGBClassifier(AbstractEstimator):
    '''
    Class that implements the xgboost classifier in random search
    to tune the tunable HPs without early stopping.
    
    Attributes
    ------------------    
    estimator_ (RandomizedSearchCV): Fitted RandomizedSearchCV instance. 
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions: dict = RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS  
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)



    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series, **kwargs) -> "MyRandomizedXGBClassifier":
        '''Fit estimator on the input data'''
        estimator = RandomizedSearchCV(
            estimator=self._create_classifier_pipeline(),
            param_distributions=add_string_to_params(self.params_distributions, "xgbclassifier__"),
            cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=self.seed),
            **SKLEARN_RANDOM_SEARCH_FIXED_PARAMS,
        )

        self.estimator_ = estimator.fit(X, y)
        return self
    

    def predict_proba(self, X: pd.DataFrame, **kwargs):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X, **kwargs)


    def _create_classifier_pipeline(self) -> Pipeline:
        '''Creates the pipeline with preprocessing + classifier'''
        if self.preprocessing == "base":
            return make_pipeline(
                VarianceThreshold(),
                XGBClassifier(**self.fixed_params)
            )
        elif self.preprocessing == "pca":
            return make_pipeline(
                VarianceThreshold(), 
                StandardScaler(),
                PCA(svd_solver="full", n_components=0.95),
                XGBClassifier(**self.fixed_params)
            )
        elif self.preprocessing == "density_filter":
            return make_pipeline(
                VarianceThreshold(),
                DensityFeatureSelector(n_target_cols=500, strategy="oversample"),
                XGBClassifier(**self.fixed_params)
            )
        else:
            raise ValueError("Unsupported preprocessing.")