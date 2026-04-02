'''
Collection of preprocessing methods commonly applied to TSS (total sum scaled) profiles,
i.e. relative abundance profiles. We implement sklearn them as sklearn transformers.
All methods return by default numpy arrays but rely on the "get_feature_names_out" sklearn
API to return pandas dataframes.
'''
import numpy as np
import pandas as pd
from typing import Literal, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array
from skbio.stats.composition import ilr
from metatab.utils.core import learn_sklearn_features_attributes
from metatab.utils.general import enlist
from metatab.utils.types import XType



class GeneralPreprocessorMixin:
    def fit(self, X: XType, y=None):
        check_array(
            X, 
            dtype="numeric", 
            ensure_2d=True, 
            ensure_all_finite=True,
            ensure_min_features=1,
            ensure_min_samples=1
        )
        for k, v in learn_sklearn_features_attributes(X).items(): 
            setattr(self, k, v)
        return self
    
    @staticmethod
    def _to_numpy(X: XType) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return X


class BaseMicrobiomeTransformer(GeneralPreprocessorMixin, TransformerMixin, BaseEstimator):
    pass


class BaseMicrobiomeSelector(GeneralPreprocessorMixin, SelectorMixin, BaseEstimator):
    pass


class LogTransformer(BaseMicrobiomeTransformer):
    '''
    Log10 of relative abundances.
    We add half of the minimun not zero value to handle zeros.
    '''
    def fit(self, X:XType, y = None):
        super().fit(X, y)
        X = self._to_numpy(X)
        self.pseudovalue_ = X[X>0].min()/2

    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        return np.log10(X + self.pseudovalue_)


class CLRTransformer(BaseMicrobiomeTransformer):
    '''
    Centered log-ratio tranformation.
    We add half of the minimun not zero value to handle zeros.
    '''
    def fit(self, X: XType, y=None):
        super().fit(X, y)
        X = self._to_numpy(X)
        self.pseudovalue_ = X[X>0].min()/2

    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        log_X = np.log(X + self.pseudovalue_)
        return log_X - log_X.mean(axis=1, keepdims=True)


class RobustCLRTransformer(BaseMicrobiomeTransformer):
    '''
    Robust centered log-ratio transformation.
    It manages zeros by excluding them from computation vs "standard "clr.
    Therefore zeros are left as they.
    '''
    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            row = X[i]
            nonzero_mask = row > 0
            log_nonzero = np.log(row[nonzero_mask])
            gm = log_nonzero.mean()
            result[i, nonzero_mask] = log_nonzero - gm
        return result


class PresenceAbsenceTransformer(BaseMicrobiomeTransformer):
    '''
    Convert relative abudances in presence/absence (1/0).
    Uses a threshold to determine presence.
    '''
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold

    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        return (X > self.threshold).astype(float)


class ArcsineTransformer(BaseMicrobiomeTransformer):
    '''
    Arcsine square root transformation.
    - scaled=False: output in [0, pi/2]
    - scaled=True:  output in [0, 1]
    '''
    def __init__(self, scaled: bool = True):
        self.scaled = scaled

    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        out = np.arcsin(np.sqrt(X))
        if self.scaled:
            out = (2 / np.pi) * out
        return out


class ILRTransformer(BaseMicrobiomeTransformer):
    '''
    Isometric log ratio transformation.
    We use the deterministic default basis implemented in scikit-bio.
    '''
    def transform(self, X: XType) -> np.ndarray:
        return ilr(self._to_numpy(X))
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        '''
        ILR transformation reduce the number of output features by 1.
        The new features are not mappable to the old ones.
        We therefore return always generic names: "ilr_{id}".
        '''
        return np.array([f"ilr_{i}" for i in range(self.n_features_in_ - 1)])


class HellingerTransformer(BaseMicrobiomeTransformer):
    '''
    Square root of relative abundance for variance stabilization.
    Returns values in [0, 1].
    '''
    def transform(self, X: XType) -> XType:
        return np.sqrt(self._to_numpy(X))
    

class FeaturePrevalenceSelector(BaseMicrobiomeSelector):
    '''
    Filter features for which their prevalence (percentage of non-zero values)
    does not satisfy the threshold.
    '''
    def __init__(self, threshold: float = 0.05):
        self.threshold=threshold

    def fit(self, X: XType, y = None) -> "FeaturePrevalenceSelector":
        _ = super().fit(X, y)
        X = self._to_numpy(X)
        self._mask = ((X > 0).sum(axis=0) / X.shape[0]) > self.threshold
        return self
    
    def _get_support_mask(self) -> np.ndarray:
        return self._mask


class RobustStandardScaler(BaseMicrobiomeTransformer):
    '''
    Robust version of sklearn StandardScaler accounting for features
    with low standard deviation due to high sparsity. A floor is added
    to the denominator (std) computed as the "std_quantile" quantile 
    of the features standard deviation distributions. This avoid inflation 
    of standardized non-zero values of high sparse features, due to
    low denominator in standardization.
    Implementation taken from SIAMCAT:
    "https://github.com/zellerlab/siamcat/blob/master/R/normalize_features.r".
    '''
    def __init__(self, std_quantile: float = 0.1):
        self.std_quantile = std_quantile

    def fit(self, X: XType, y = None) -> "RobustStandardScaler":
        super().fit(X, y)
        X = self._to_numpy(X)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.q_std_ = np.quantile(X.std(axis=0), self.std_quantile)
        self.std_ = X.std(axis=0, mean=self.mean_, keepdims=True) + self.q_std_       
        return self

    def transform(self, X: XType) -> np.ndarray:
        X = self._to_numpy(X)
        return (X - self.mean_) / self.std_



class FeatureDensitySelector(BaseMicrobiomeSelector):
    '''
   Selects the N most dense columns. Some key features:
    - The selector can select a number of columns different from the desired target,
    depending on the selection strategy used.
    - The selector can exclude all features. In this case is possible to fine-control
    its behaviour through the "on_empty" parameter.

    Parameters:
        n_target_cols (int, optional):
            Desired number of columns. Must be a integer in [0, inf].
            If 0 all columns are filtered, if inf all columns are kept.

        strategy (Literal["exact", "oversample", "undersample"], optional):
            - exact: select exactly "n_target_cols" columns. 
            The ties are arbitrarily broken, even though the results are consistent with a fixed input.
            - oversample: include all ties on the boundary, resulting possibly in more than "n_target_cols".
            - undersample: exclude all ties on the boundary only if this means overshooting the target number,
            resulting in less than "n_target_cols".

        on_empty (Literal["select_all", "error"], optional):
            Set the transformer behaviour when all columns are filtered:
            - error: raise an error.
            - select_all: Suppress the transformer action, meaning all columns are selected.

    ## Attributes:
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
        n_target_cols: int = 500,
        strategy: Literal["exact", "oversample", "undersample"] = "exact",
        on_empty: Literal["select_all", "error"] = "select_all"
    ):
        self.n_target_cols = n_target_cols
        self.strategy = strategy
        self.on_empty = on_empty


    def fit(self, X: XType, y = None) -> "FeatureDensitySelector":        
        super().fit(X, y)
        
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
        return self._mask
    


PREPROCESSING_REGISTRY = {
    "zero_variance": VarianceThreshold,
    "log": LogTransformer,
    "clr": CLRTransformer,
    "rclr": RobustCLRTransformer,
    "ilr": ILRTransformer,
    "pa": PresenceAbsenceTransformer,
    "arcsin": ArcsineTransformer,
    "hellinger": HellingerTransformer,
    "prevalence": FeaturePrevalenceSelector,
    "density": FeatureDensitySelector,
    "rsc": RobustStandardScaler
}


PreprocessingStrategy = Literal[
    "zero_variance",
    "log",
    "clr",
    "rclr",
    "ilr",
    "pa",
    "arcsin",
    "hellinger",
    "prevalence",
    "density",
    "rsc"
]


def build_preprocessing_pipeline(preprocessing: PreprocessingStrategy | list[PreprocessingStrategy]) -> Pipeline:
    '''
    Build the preprocessing pipeline.
    Takes in input a single or list of strategies.
    The preprocessing steps use always their default implementation.
    '''
    preprocessings = enlist(preprocessing)
    selected_prep = []
    for prep in preprocessings:
        prep_class = PREPROCESSING_REGISTRY[prep]
        selected_prep.append([prep, prep_class()])
    return Pipeline(selected_prep)