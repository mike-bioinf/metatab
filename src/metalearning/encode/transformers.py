import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from metatab_utils.general import enlist



class NanToNone(TransformerMixin, BaseEstimator):
    '''
    Scikit-like transformer to convert nan to None.
    Can be fitted on pandas DataFrames only (no numpy arrays).

    Parameters:
        columns (str | list[str]): 
            Columns on which apply the transformation.
        check_on_fit (bool): 
            Whether execute the data checks at fit level.
            In detail `X` is checked to be a pandas DataFrame,
            and `columns` is checked to be present in it.
            Note that the X data passed in fit is just required for pipeline compability. 
            The transformer uses and acts only the `X` passed in transform.

    '''
    def __init__(self, columns: str | list[str], check_on_fit: bool):
        self.columns = columns
        self.check_on_fit = check_on_fit

    def fit(self, X: pd.DataFrame, y: None = None) -> "NanToNone":
        self._list_columns = enlist(self.columns)
        if self.check_on_fit:
            _check_dataframe_type(X)
            _check_columns_presence(X, self._list_columns)
        return self
    
    def transform(self, X: pd.DataFrame, y: None = None) -> pd.DataFrame:
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        X_copy = deepcopy(X)
        X_copy[self._list_columns] = X_copy[self._list_columns].replace({np.nan: None})
        return X_copy.to_numpy()

    def get_feature_names_out(self, input_features = None) -> np.ndarray[str]:
        if not hasattr(self, "feature_names_in_"):
            return np.array([f"col_{i}" for i in self.n_features_in_])
        return self.feature_names_in_



class ColToStr(TransformerMixin, BaseEstimator):
    '''
    Scikit-like transformer casting DataFrame columns 
    to object-dtyped columns and column values to str type.
    Works only on DataFrame (no numpy arrays).

    Parameters:
        columns (str | list[str]): 
            Columns to transform.
        check_on_fit (bool): 
            Whether execute the data checks at fit level.
            In detail `X` is checked to be a pandas DataFrame,
            and `columns` is checked to be present in it.
            Note that the X data passed in fit is just required for pipeline compability. 
            The transformer uses and acts only the `X` passed in transform.
    '''
    def __init__(self, columns: str | list[str], check_on_fit: bool):
        self.columns = columns
        self.check_on_fit = check_on_fit

    def fit(self, X: pd.DataFrame, y: None = None) -> "ColToStr":
        self._list_columns = enlist(self.columns)
        if self.check_on_fit:
            _check_dataframe_type(X)
            _check_columns_presence(X, self._list_columns)
        return self

    def transform(self, X: pd.DataFrame, y: None = None) -> pd.DataFrame:
        _check_dataframe_type(X)
        _check_columns_presence(X, self._list_columns)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        X_copy = deepcopy(X)
        X_copy = X_copy.astype({col: "str" for col in self._list_columns})
        return X_copy.to_numpy()

    def get_feature_names_out(self, input_features = None) -> np.ndarray[str]:
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