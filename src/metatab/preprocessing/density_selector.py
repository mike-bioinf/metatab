import numpy as np
import pandas as pd
from typing import Literal, Any
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from metatab.utils.types import XType



class DensityFeatureSelector(SelectorMixin, BaseEstimator):
    '''
    Sklearn-like transformer that selects the N most dense columns. Some key features:
    - The selector expect non-empty, full numeric dataframes/arrays without missing values.
    - The selector can select a number of columns different from the desired target,
    depending on the selection strategy used.
    - The selector can exclude all features. In this case is possible to fine-control
    its behaviour through the "on_empty" parameter.

    Parameters:
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

        on_empty (Literal["select_all", "error"], optional):
            Set the transformer behaviour when all columns are filtered:
            - error: raise an error.
            - select_all: Suppress the transformer action, meaning all columns are selected.

    ## Attributes
        strategy_ (str): 
            Selection strategy used.

        n_target_features_ (int):
            Target number of columns.

        n_features_in_ (int): 
            Number of columns seen at fit level.
        
        feature_names_in_ (np.ndarray): 
            Column names seen at fit level. Set only if X is a dataframe.

        n_selected_features_ (int):
            Number of selected features.

        densities_ (pd.Series):
            Density scores for the columns seen at fit level.

        minimum_density_score_ (float):
            The smallest density score among the selected features.
    '''
    def __init__(
        self, 
        n_target_cols: int,
        strategy: Literal["exact", "oversample", "undersample"],
        on_empty: Literal["select_all", "error"] = "select_all"
    ):
        self.n_target_cols = n_target_cols
        self.strategy = strategy
        self.on_empty = on_empty


    def fit(self, X: XType, y = None) -> "DensityFeatureSelector":
        '''
        Fit the selector on the input data obtaining the columns mask. 
        This is an boolean array indicating which features has to be 
        selected and which not based on position.
        '''
        check_array(
            X, 
            dtype="numeric", 
            ensure_2d=True, 
            ensure_all_finite=True,
            ensure_min_features=1,
            ensure_min_samples=1
        )
        
        if isinstance(X, pd.DataFrame) and all([isinstance(col, str) for col in X.columns]):
            self.feature_names_in_ = np.array(X.columns)
        
        # create mock dataframe from numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # densities series mantains columns names
        self.densities_ = (X != 0).mean(axis=0)

        features_to_keep, minimum_density_score = self._get_indexes_to_retain(
            self.densities_,
            self.n_target_cols,
            self.strategy
        )

        self.strategy_ = self.strategy
        self.n_target_features_ = self.n_target_cols
        self.n_features_in_ = X.shape[1]

        if not features_to_keep:
            if self.on_empty == "error":
                raise ValueError(
                    "Feature selection resulted in an empty feature set."
                )
            elif self.on_empty == "select_all":
                self.n_selected_features_ = self.n_features_in_
                self._mask = np.array([True] * self.n_features_in_)
                # we allow made-up names since it's only for set_output API
                self._selected_features = X.columns.to_numpy()
                # when we pass all features then the minimun density score is the maximum
                self.minimum_density_score_ = self.densities_.max()
            else:
                raise ValueError(
                    "on_empty can be only set to 'select_all' or 'error'."
                )
        else:
            self.n_selected_features_ = len(features_to_keep)
            self._mask = X.columns.isin(features_to_keep)
            self._selected_features = X.columns.to_numpy()[self._mask]
            self.minimum_density_score_ = minimum_density_score
        
        return self

    
    @staticmethod
    def _get_indexes_to_retain(
        densities: pd.Series, 
        n_target: int, 
        strategy: Literal["exact", "oversample", "undersample"]
    ) -> tuple[list[Any], float]:
        '''
        Get the list of indexes to retain to reach the target number of elements.
        The selection is guided by the density scores, i.e. only the n_target 
        most dense columns are kept.A number of indexes different than n_target 
        can be returned depending on the strategy. 
        A void list can be returned in some cases, for example when n_target is 0 
        or with the 'undersample' strategy.

        Parameters:
            densities (pd.Series):
                Series of density values.
            
            n_target (int):
                Target number of elements to retain. Must be a integer in [0, inf].
                If 0 an empty list is returned, if inf all columns index are returned.
            
            strategy (Literal["exact", "oversample", "undersample"]):
                - exact: keep exactly n_target elements. The ties are arbitrarily broken,
                even though the results are consistent with a fixed input.
                - oversample: include all ties on the boundary.
                - undersample: exclude all ties on the boundary if this prevents overshooting 
                the target number, otherwise keep them. 

        Returns:
            tuple[list,float]:
            The indexes to keep as a list. The list can be void.
            The minimum density score that is kept. Equal to -1 if no index is kept.
        '''
        if n_target < 0:
            raise ValueError("n_target must be in [0, inf].")
        
        if n_target == 0:
            # we use -1 to indicate that the minimum density score is not determinable 
            return [], -1.0
        
        # use stable algorithm to get reproducible order
        sorted_densities = densities.sort_values(ascending=False, kind="stable")

        if densities.size <= n_target:
            return densities.index.to_list(), sorted_densities.iloc[-1]
        
        if strategy == "exact":
            return (
                sorted_densities.iloc[:n_target].index.to_list(),
                sorted_densities.iloc[n_target-1]
            )
        
        elif strategy == "oversample":
            target_density = sorted_densities.iloc[n_target-1]
            return (
                sorted_densities[sorted_densities >= target_density].index.to_list(),
                target_density
            )
        
        elif strategy == "undersample":
            target_density = sorted_densities.iloc[n_target-1]
            right_densities = sorted_densities.iloc[n_target:]
            n_right_ties = (right_densities == target_density).sum()

            if n_right_ties == 0:
                # we keep the element on the boundary
                return (
                    sorted_densities[sorted_densities >= target_density].index.to_list(),
                    target_density
                )
            else:
                # we exclude all ties on the boundary
                indexes = sorted_densities[sorted_densities > target_density].index.to_list()
                target_density = target_density if indexes else -1
                return indexes, target_density
        
        else:
            raise ValueError("strategy must be one of 'exact', 'oversample' or 'undersample'.")
        

    def _get_support_mask(self) -> np.ndarray:
        '''Retrieve the fitted feature mask'''
        check_is_fitted(self, "_mask")
        return self._mask


    def get_feature_names_out(self, input_features = None) -> np.ndarray:
        # when fitted on numpy array the resulting names and order is irrelevant
        if not hasattr(self, "feature_names_in_"):
            return np.array([f"col_{i}" for i in self.n_features_in_])
        return self._selected_features