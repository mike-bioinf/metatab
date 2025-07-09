import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from fit.filtering.percentage import get_filtering_thresh, filter_percentage



class SparseFeaturesReductor(SelectorMixin, BaseEstimator):
    '''
    Reduce the number of columns to/under a target value guided by sparsity.
    The filtering is sparsity-based in the sense that the most sparse
    features are removed to reach the target number of columns.

    The selector treats all numerical columns as potentially filterable.
    In detail the columns selected by the `select_dtypes("number")` call are 
    considered filterable. No difference is therefore made between float
    and integer typed columns. 

    The algorithm can reduce the columns number under the target value.
    This is because multiple columns can share the same percentage of non-zero
    values. In this case the algorithm is designed to filter all of them if some
    of them should be removed to to reach the target value. 
    No random selection is performed on them in order to exactly reach 
    the target value.

    The selector is designed to NOT filter full dense columns.
    Therefore sometimes is not possible to reach the desired number of columns,
    for example when the number of fully dense columns is greater then the desired.

    In this scenarios the behaviour of the class can be controlled with the
    "error_on_fail" parameter. Briefly is this is set to True then an error
    is raised, otherwise the input unfiltered dataframe is returned.


    Parameters
    -----------------
    n_cols_target (int):
        Desired number of columns.
    
    na_as_zero (bool, optional):
        Whether to treat nan, NA, None values as 0 in filtering procedures.
        If False they are not considered in percentage computation, 
        meaning the column is treated as a "shortened" version.

    error_on_fail (bool, optional):
        Whether to raise an error if the reduction is not possible,
        i.e. all numeric columns are full dense.


    Attributes
    -----------------
    n_features_in_ (int): 
        Number of columns at fit level.
    
    feature_names_in_ (np.ndarray): 
        Column names array at fit level. 
        Set only if fit receive a DataFrame and 
        if the column index  has a single level.
    '''
    def __init__(self, n_cols_target: int, na_as_zero: bool = True, error_on_fail: bool = False):
        self.n_cols_target = n_cols_target
        self.na_as_zero = na_as_zero
        self.error_on_fail = error_on_fail


    def fit(self, X: pd.DataFrame | np.ndarray, y = None) -> "SparseFeaturesReductor":
        '''
        Fit the selector on the input data obtaining the 
        logical columns mask. This is an boolean array indicating 
        which features has to be selected and which not.
        Sets the "features" attrs needed for checking.
        The y parameter is ignored (present for compability). 
        '''
        check_array(X, ensure_2d=True, ensure_all_finite="allow-nan")
        self.n_features_in_ = X.shape[1]
    
        if isinstance(X, pd.DataFrame) and self._are_all_columns_string(X):
            self.feature_names_in_ = np.array(X.columns)
        
        # creating surrogate column names for numpy arrays
        if isinstance(X, np.ndarray):
            columns = [f"col_{i}" for i in range(self.n_features_in_)]
            X = pd.DataFrame(X, columns=columns)
        
        thresh = get_filtering_thresh(
            df=X, 
            target_ncols=self.n_cols_target,
            zero_na=self.na_as_zero,
            quiet=True
        )

        if thresh is None and self.error_on_fail:
            raise ValueError(
                f"Is not possible to reach the target number of columns of {self.n_cols_target}."
            )
        
        X_fit_filtered = filter_percentage(
            df=X,
            threshold_percent=thresh,
            zero_na=self.na_as_zero
        )
        
        mask = []
        for col in X.columns:
            if col in X_fit_filtered.columns:
                mask.append(True)
            else:
                mask.append(False)
        
        self._mask = np.array(mask)
        return self


    def _get_support_mask(self) -> np.ndarray:
        '''Retrieve the fitted features mask'''
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