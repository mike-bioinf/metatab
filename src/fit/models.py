import numpy as np
import pandas as pd
from typing import Literal
from abc import ABC, abstractmethod
from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from fit.random_search import MyRandomSearchCV

from fit.constants import (
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS, 
    RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
)



class AbstractEstimator(ABC):
    '''Blueprint for estimators classes'''

    @abstractmethod
    def __init__(
        self, 
        preprocessing: Literal["no", "sparse_filter", "pca"],
        seed: int,
        params_distributions: dict,
        fixed_params: dict
    ):
        pass

    @abstractmethod
    def fit(*args, **kwargs):
        pass

    @abstractmethod
    def predict_proba(*args, **kwargs):
        pass
    
    @abstractmethod
    def _create_preprocessing_pipeline(self) -> None | Pipeline:
        pass



class MyESRandomizedXGBClassifier(AbstractEstimator):
    
    def __init__(
        self, 
        preprocessing: Literal["no", "sparse_filter", "pca"],
        seed: int,
        params_distributions: dict = RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
    ):
        '''
        Class that uses a custom implementation of RandomSearchCV (MyRandomSearchCV)
        with the XGBClassifier.
        This custom implementation allows and enforces the use of early stopping
        during the ensemble building process.
        Allows to use different features preprocessing strategies.
        
        Parameters
        ----------------
        preprocessing (Literal["no", "sparse_filter", "pca"]): 
            Preprocessing strategy to use.
        seed (int): 
            Seed for reproducibility.
        params_distributions (dict):
            Dict of param:distributions from which to sample values in the tuning process.
        fixed_params (dict):
            Dict of param:value that are fixed i.e. not tuned in the search.

        Attributes
        ------------------    
        estimator_ (MyRandomSearchCV): Fitted MyRandomSearchCV instance. 
        '''
        self.preprocessing = preprocessing
        self.seed = seed
        self.params_distributions = params_distributions
        self.fixed_params = fixed_params


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

        estimator.fit(X, y)
        self.estimator_ = estimator
        return self
    

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        '''Predict probabilities on X test'''
        return self.estimator_.predict_proba(X, **kwargs)
        
        
    def _create_preprocessing_pipeline(self) -> Pipeline:
        '''
        Create the preprocessing pipeline based on the "preprocessing".
        Returns a Pipeline object.
        '''
        if self.preprocessing == "no":
            return make_pipeline(VarianceThreshold())
        elif self.preprocessing == "pca":
            return make_pipeline(
                VarianceThreshold(), 
                StandardScaler(),
                PCA(svd_solver="full", n_components=0.95)
            )
        elif self.preprocessing == "sparse_filter":
            ## TODO: we have to implement the filtering procedure as sklearn transformer
            pass
        else:
            raise ValueError("Unsupported preprocessing.")