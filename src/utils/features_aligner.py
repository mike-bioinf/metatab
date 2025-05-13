import numpy as np
import pandas as pd
from typing import Literal, Any
from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array



class FeaturesAligner(TransformerMixin, BaseEstimator):
    '''
    This class implements a sklearn compatible transformer that reindex data.
    So the class learns the feature in fit and reindex the test data in transform.
    
    The class works only with pandas dataframes with an column index of all strings.
    In detail when these conditions are not met by the array passed in fit,
    the class will skip the transformation step (does nothing).
    The tranform method will therefore return the input X transforming it to a pandas 
    DataFrame or not (see 'convert_on_skip' parameter).
    Instead when the class has learned the transformation but the conditions are not 
    respected by the array passed in transform, then an error is raised.
    
    Parameters:
        fill_value (Any, optional):
            Value used to fill the columns created in transform. Deafaults to 0.0.

        raise_on_edge (Literal["warning", "error"], optional): 
            Whether to raise a warning or an error when the dataframe passed in transform
            has no column in common with the learned ones.
            If "warning" then the transformation will produce a dataframe full
            o new columns of fill_value values.
            Deafults to "error".

        convert_on_skip (bool, optional):
            Whether to convert to a pandas DataFrame the data when the transformation is skipped.
            See class description for details about the skipping logic.
            Can be overriden by set_output instruction.

    Attributes
        feature_names_in_ (np.array[str]): Names of columns learned in fit.
        n_features_in (int): Number of features learned in fit.
    '''

    def __init__(
        self, 
        fill_value: Any = 0.0, 
        raise_on_edge: Literal["warning", "error"] = "error",
        convert_on_skip: bool = True
    ):
        self.fill_value = fill_value
        self.raise_on_edge = raise_on_edge
        self.convert_on_skip = convert_on_skip  



    def fit(self, X: pd.DataFrame | np.ndarray, y = None) -> 'FeaturesAligner':
        '''
        Parameters:
            X (pd.DataFrame | np.ndarray): 
                DataFrame on which the features are learned.
                If a numpy array nothing is done.
            y: Ignored.
        '''
        check_array(X, ensure_all_finite=False, dtype=None)

        if isinstance(X, pd.DataFrame) and self._are_all_columns_string(X):
            self._skip_transform = False
            self.feature_names_in_ = np.array(X.columns)
            self.n_features_in_ = X.shape[1]
        else:
            self.feature_names_in_ = None
            self._skip_transform = True
        
        return self



    def transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        '''
        Align X to the feature space learned during fit.

        Parameters:
            X (pd.DataFrame): DataFrame to reindex.
            y: Ignored.
        
        Returns:
            pd.DataFrame: The transformed X. Note that depending on 
            convert_on_skip (initialization parameter) the returned object can also
            NOT be a pandas DataFrame.
        '''
        check_is_fitted(self, attributes=["_skip_transform"])
        check_array(X, ensure_all_finite=False, dtype=None)

        if self._skip_transform:
           return (pd.DataFrame(X) if self.convert_on_skip else X)
        elif not self._skip_transform and not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        elif not self._skip_transform and not self._are_all_columns_string(X):
            raise KeyError("All X columns must be strings.")
        
        # raise condition if X has no learned columns
        if not (set(X.columns) & set(self.feature_names_in_)):
            self._raise_condition(
                f"X has no learned columns! A dataframe full of {self.fill_value} (fill_value) is produced.",
                self.raise_on_edge
            )
        
        new_X = X.reindex(columns=self.feature_names_in_, fill_value=self.fill_value)
        return new_X

    

    def get_feature_names_out(self):
        '''
        Method that enables the set_output utility.
        Must return the learned column names.
        '''
        return self.feature_names_in_


    @ staticmethod
    def _are_all_columns_string(X: pd.DataFrame):
        all_strings = True

        for col in X.columns:
            if not isinstance(col, str):
                all_strings = False
                break
        
        return all_strings


    @staticmethod
    def _raise_condition(message: str, what: Literal["warning", "error"]) -> None:
        if what == "warning":
            warn(message)
        else:
            raise ValueError(message)