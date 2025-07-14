import numpy as np
import pandas as pd
from typing import Literal
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from estimators.preprocessing.utils import get_density_scores, get_index_to_retain



class DensityFeatureSelector(SelectorMixin, BaseEstimator):
    '''
    Select the most N dense columns.

    The selector expect non-empty, full numeric dataframes/arrays without missing values.

    The selector can select a number of columns different from the desired target,
    depending on the selection strategy used.

    The selector can exclude all features. In this case is possible 
    to raise an error or allow this scenario through the "error_on_empty" flag.


    Parameters
    -----------------
    n_target_cols (int):
        Desired number of columns. Must be a integer in [0, inf].
        If 0 all columns are filtered, if inf all columns are kept.

     strategy (Literal["exact", "oversample", "undersample"]):
        - exact: select exactly "n_target_cols" columns. 
        The ties are arbitrarily broken, even though the results 
        are consistent with a fixed input.
        - oversample: include all ties on the boundary, 
        resulting possibly in more than "n_target_cols".
        - undersample: exclude all ties on the boundary 
        only if this means overshooting the target number,
        resulting in less than "n_target_cols".

    error_on_empty (bool, optional):
        Whether to raise an error if the selector excludes all features.


    Attributes
    -----------------
    n_features_in_ (int): 
        Number of columns seen at fit level.
    
    feature_names_in_ (np.ndarray): 
        Column names seen at fit level.
        Set only if X is a dataframe.

    densities_ (pd.Series):
        Density scores for the columns seen at fit level.
    '''
    def __init__(
        self, 
        n_target_cols: int,
        strategy: Literal["exact", "oversample", "undersample"],
        error_on_empty: bool = False
    ):
        self.n_target_cols = n_target_cols
        self.strategy = strategy
        self.error_on_empty = error_on_empty


    def fit(self, X: pd.DataFrame | np.ndarray, y = None) -> "DensityFeatureSelector":
        '''
        Fit the selector on the input data obtaining the columns mask. 
        This is an boolean array indicating which features has to be selected 
        and which not based on position.
        Sets the "feature*" attrs needed for checking.
        The y parameter is ignored (present for compability). 
        '''
        check_array(
            X, 
            dtype="numeric", 
            ensure_2d=True, 
            ensure_all_finite=True,
            ensure_min_features=1,
            ensure_min_samples=1
        )
        
        self.n_features_in_ = X.shape[1]
    
        if isinstance(X, pd.DataFrame) and self._are_all_columns_string(X):
            self.feature_names_in_ = np.array(X.columns)
        
        # create mock dataframe from numpy array
        if isinstance(X, np.ndarray):
            columns = [f"col_{i}" for i in range(self.n_features_in_)]
            X = pd.DataFrame(X, columns=columns)
        
        self.densities_ = get_density_scores(X)
        cols_to_keep = get_index_to_retain(self.densities_, self.n_target_cols, self.strategy)

        if not cols_to_keep and self.error_on_empty:
            raise ValueError("Feature selection resulted in an empty feature set.")
        
        self._mask = X.columns.isin(cols_to_keep)
        return self


    def _get_support_mask(self) -> np.ndarray:
        '''Retrieve the fitted feature mask'''
        check_is_fitted(self, "_mask")
        return self._mask


    @staticmethod
    def _are_all_columns_string(X: pd.DataFrame):
        all_strings = True

        for col in X.columns:
            if not isinstance(col, str):
                all_strings = False
                break
        
        return all_strings