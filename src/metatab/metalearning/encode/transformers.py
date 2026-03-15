import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from metatab.utils.general import enlist
from metatab.utils.core import check_predict_features



class NanToNone(TransformerMixin, BaseEstimator):
    '''
    Scikit-like transformer to convert nan to None.
    Can be fitted on pandas DataFrames only (no numpy arrays).

    Parameters:
        columns (str | list[str]):
            Columns on which apply the transformation.
    '''
    def __init__(self, columns: str | list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> "NanToNone":
        self._list_columns = enlist(self.columns)
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        self.n_features_in_ = X.shape[1]
        learn_feature_names_in(self, X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        check_predict_features(self, X)
        X_copy = deepcopy(X)
        # to avoid automatic downcasting with replace
        with pd.option_context('future.no_silent_downcasting', True):
            X_copy[self._list_columns] = X_copy[self._list_columns].replace({np.nan: None})
        return X_copy.to_numpy()

    def get_feature_names_out(self, input_features = None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            return np.array([f"col_{i}" for i in self.n_features_in_])
        return self.feature_names_in_



class ColToStr(TransformerMixin, BaseEstimator):
    '''
    Scikit-like transformer casting DataFrame columns 
    to object-dtyped columns and column values to str type.
    Works only on DataFrame (no numpy arrays).

    Parameters:
        columns (str | list[str]): Columns to transform.
    '''
    def __init__(self, columns: str | list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> "ColToStr":
        self._list_columns = enlist(self.columns)
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        self.n_features_in_ = X.shape[1]
        learn_feature_names_in(self, X)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        check_predict_features(self, X)
        X_copy = deepcopy(X)
        X_copy = X_copy.astype({col: "str" for col in self._list_columns})
        return X_copy.to_numpy()

    def get_feature_names_out(self, input_features = None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            return np.array([f"col_{i}" for i in self.n_features_in_])
        return self.feature_names_in_
    


class InfToNan(TransformerMixin, BaseEstimator):
    '''
    Scikit-like transformer converting +/-inf values to nan.
    Can be fitted on DataFrame only (no numpy arrays).
    '''
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: None = None) -> "InfToNan":
        _check_dataframe_type(X)
        self.n_features_in_ = X.shape[1]
        learn_feature_names_in(self, X)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        _check_dataframe_type(X)
        check_predict_features(self, X)
        X_copy = deepcopy(X)
        # to avoid automatic downcasting with replace
        with pd.option_context('future.no_silent_downcasting', True):
            return X_copy.replace(to_replace=[np.inf, -np.inf], value=np.nan).to_numpy()
    
    def get_feature_names_out(self, input_features = None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            return np.array([f"col_{i}" for i in self.n_features_in_])
        return self.feature_names_in_
    


def _check_columns_presence(X: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in X.columns:
            raise ValueError(f"'{col}' column not found in X.")


def _check_dataframe_type(X: Any) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    

def learn_feature_names_in(obj: Any, X: pd.DataFrame) -> None:
    cols = X.columns
    if all([isinstance(col, str) for col in cols]):
        obj.feature_names_in_ = cols.to_numpy()