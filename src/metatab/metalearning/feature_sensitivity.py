from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Literal
from metatab.utils.general import select_level_from_columns, ensure_or_create

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.metalearning.surrogate_rf import SurrogateRandomForestRegressor



def compute_feature_sensitivity_map(
    model: SurrogateRandomForestRegressor | Pipeline, 
    X: pd.DataFrame,
    column_level_groups: str | int,
    column_level_prediction: int | str | None = -1,
    exclude_groups: str | list[str] | None = None,
    n_permutations: int = 20, 
    seed: int = 0
) -> dict[str, np.ndarray]:
    '''
    Compute model sensitivity to groups of features using a grouped permutation approach.

    Parameters:
        model (SurrogateRandomForestRegressor | Pipeline):
            Trained surrogate model or pipeline on which compute feature sensitivity.
        
        X (pd.DataFrame): 
            Data on which to evaluate feature sensitivity. 
            Columns must be a MultiIndex.
        
        column_level_groups (str | int): 
            Level name or index of the column MultiIndex used to group features.
            Sensitivity is computed for each unique value at this level.

        column_level_prediction (int | str | None, optional):
            X columns level to select for model predictions. 
            Models and pipelines cannot be adapted to manage the MultiIndex. 
            Therefore one can select the single level to keep for model inference.

        exclude_groups (str | list[str] | None, optional):
            Groups to exclude from the permutation procedure.
            We assume that the index are strings since this is our scenario.

        n_permutations (int, optional):
            Number of random permutations to generate for sensitivity estimation.

        seed (int, optional):
            Seed controlling permutation reproducibility.

    Returns:
        dict[str,np.ndarray]:
        Dictionary mapping each selected group of the "column_level_group"
        to an array of absolute mean prediction differences across permutations.
    '''
    X_pred = select_level_from_columns(X, column_level_prediction) \
        if column_level_prediction \
        else X
        
    ori_pred, _ = model.predict(X_pred)
    index_permutations = _create_index_permutations(X, n_permutations, seed)
    map_feature_sensivity = {}

    categories = X.columns.unique(column_level_groups)
    selected_categories = categories[~categories.isin(ensure_or_create(exclude_groups, list))]

    for category in selected_categories:
        sensitivity = []
        for index_permutation in index_permutations:
            X_permuted = _permute_block(X, index_permutation, column_level_groups, category)

            X_permuted_pred = select_level_from_columns(X_permuted, column_level_prediction) \
                if column_level_prediction \
                else X_permuted

            perm_pred, _ = model.predict(X_permuted_pred)
            sensitivity.append(np.mean(np.abs(ori_pred - perm_pred)))
        map_feature_sensivity[category] = np.array(sensitivity)

    return map_feature_sensivity



def _permute_block(
    X: pd.DataFrame,
    index_permutation: np.ndarray,
    column_level_groups: int | str, 
    column_level_category: str,
    raise_on_empty_permutation: Literal["none", "warning", "error"] = "error"
) -> pd.DataFrame:
    '''
    Permute a block of columns in a DataFrame.
    The block to permute is identified by a specific level (or name) of the column MultiIndex
    and the corresponding category value. The permutation is applied to these columns only, 
    following the provided index order. All columns within the selected block
    share the same permutation, ensuring a homogeneous swap between groups of features
    across different samples.
    '''
    X_permuted = X.copy()

    # select the columns that belong to the chosen block
    mask = X.columns.get_level_values(column_level_groups) == column_level_category
    
    # manage case in which no column is selected
    if not mask.any():
        message = "No columns have been selected based on 'column_level_groups' and 'column_level_category'."
        if raise_on_empty_permutation == "error":
            raise ValueError(message)
        elif raise_on_empty_permutation == "warning":
            warnings.warn(message)
        elif raise_on_empty_permutation == "none":
            return X_permuted
        else:
            raise ValueError("Unsupported 'raise_on_empty_permutation' value.")
    
    X_permuted.loc[:, mask] = X.iloc[index_permutation, mask].to_numpy()
    return X_permuted



def _create_index_permutations(X: pd.DataFrame, n_permutations: int, seed: int) -> list[np.ndarray]:
    '''Creates and returns a list of index row-permutations.'''
    rng = np.random.default_rng(seed)
    return [rng.permutation(X.shape[0]) for _ in range(n_permutations)]