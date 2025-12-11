import numpy as np
import pandas as pd
from dataclasses import dataclass
from metatab_utils.types import XType, YType


@dataclass
class BagCV:
    '''
    Cross-validated bagging Dataclass.
    Parameters:
        n_repeats (int): Number of cv repeats
        n_folds (int): Number of cv folds
        seed (int): Cv seed
    '''
    n_repeats: int
    n_folds: int
    seed: int


def collect_sklearn_classification_fit_info_from_data(X: XType, y: YType) -> dict:
    '''
    Collect the tipical sklearn classification info from the fit data.
    In detail we derive the `classes_`, `n_features_in_` and when
    possible the `feature_names_in_` info using these string as keys. 
    '''
    y = y.to_numpy() if isinstance(y, pd.Series) else y
    res = {"classes_": np.unique(y), "n_features_in_": X.shape[1]}

    if isinstance(X, pd.DataFrame) and all([isinstance(col, str) for col in X.columns]):
        feature_names_in = X.columns
        res["feature_names_in_"] = feature_names_in
    
    return res