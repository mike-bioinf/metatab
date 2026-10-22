from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Literal
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from metatab.estimators.utils.general import check_predict_features
from metatab.estimators.core.configurations import EarlyStopConfiguration
from metatab.estimators.utils.general import learn_sklearn_features_attributes

if TYPE_CHECKING:
    from metatab.estimators.estimators import DefaultEstimator
    from metatab.preprocessing import PreprocessingStrategy
    from metatab.metatab_utils.types import XType, YType



class BaseDefault(ClassifierMixin, BaseEstimator):
    '''
    Parameters:
        preprocessing (PreprocessingStrategy, optional):
            Preprocessing strategy to apply.
        
        seed (int, optional):
            Random seed controlling classifier randomness and the inner cross-validation procedure.

        n_threads (int, optional):
            Number of threads used to parallelize classifier fitting. 
            The hyperparameter search itself is not parallelized to avoid process overload.
            In general is better to parallelize either model fitting or search, but not both.

        device (Literal["cpu", "cuda", "auto"], optional):
            Device to fit the model(s) on.
            - "cpu" or "cuda" explicitly selects the device.
            - "auto" uses GPU if available and supported by the classifier; otherwise CPU.
            
    ## Attributes:

        classes_ (np.ndarray): 
            The array of class labels learnt at fit time.

        n_features_in_ (int): 
            Number of features seen during fit.
        
        feature_names_in_ (np.ndarray):
            Names of features seen during fit.
            Exists only when fitted on a pandas DataFrame with strings as columns.
    '''
    def __init__(
        self,
        preprocessing: PreprocessingStrategy = "base",
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto"
    ):
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_threads=n_threads
        self.device=device


    def fit(
        self,
        X: XType,
        y: YType,
        early_stop_configuration: None | EarlyStopConfiguration = None
    ):
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity
        
        estimator: DefaultEstimator = self.myclass(
            preprocessing=self.preprocessing,
            vary_preprocessing=False,
            seed=self.seed,
            n_threads=self.n_threads,
            device=self.device,
            early_stop_configuration=early_stop_configuration
        )

        self.estimator_ = estimator.fit(X, y)
        self.classes_ = le.classes_
        for k, v in learn_sklearn_features_attributes(X).items():
            setattr(self, k, v)
        return self
    

    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict class for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            np.ndarray: The predicted classes.
        '''
        check_is_fitted(self, "estimator_")
        check_predict_features(self, X)
        return self.estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.
        
        Returns:
            np.ndarray: The class probabilities of the input samples.
        '''
        check_is_fitted(self, "estimator_")
        check_predict_features(self, X)
        return self.estimator_.predict_proba(X) 