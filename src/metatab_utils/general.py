from __future__ import annotations

import pandas as pd
from copy import deepcopy
from typing import Any, Callable, TYPE_CHECKING
from dataclasses import fields, is_dataclass

if TYPE_CHECKING:
    import numpy as np
    from metatab_utils.types import XType, YType



def add_broadcasted_objects_as_column(
    df: pd.DataFrame, 
    dictionary: dict[Any, Any],
    convert_bool_to_str: bool = False,
    convert_none_to_str: bool = False,
    force_object_datatype: list = [],
    check_matching_keys_cols: bool = True,
    check_non_builtin_types: bool = True,
    allowed_non_builtin_types: tuple = tuple(),
    copy: bool = True 
) -> pd.DataFrame:
    '''
    Utility to add and broadcast complex objects into dataframe columns.
    The new columns will contain repetition of the same object regardless of its type.

    Parameters:
        df (pd.DataFrame): Dataframe.

        dictionary (dict[Any, Any]): Dict of objects.

        convert_bool_to_str (bool, optional):
            Whether to convert booleans to their string representation.

        convert_none_to_str (bool, optional):
            Whether to convert None object to "None" (string representation).
        
        force_object_datatype (list, optional):
            A list of the new column names (dict keys) to be forced to "object" datatype.
            Note that this applies only to the columns created starting
            from the dict keys, both new and overwritten ones. 

        check_matching_keys_cols (bool, optional): 
            Whether to check if the dictionary keys match the dataframe column index. 
            The match is checked via the equality operator ("==").
            IF True an error is raised when the condition is met.
        
        check_non_builtin_types (bool, optional):
            Whether to check for non builtin objects in the dict.
            If True an error is raised when the condition is met.
        
        allowed_non_builtin_types (tuple, optional): 
            Tuple of non builtin types not triggering the builtin type check.
            Ignored if "check_non_builtin_types" is False.
        
        copy (bool, optional):
            Whether to work and return a copy of the input dataframe.

    Returns:
        pd.DataFrame: The new/updated dataframe.
    '''
    df = deepcopy(df) if copy else df

    if check_matching_keys_cols:
        for col in df.columns:
            if col in dictionary.keys():
                raise ValueError(f"'{col}' triggers a key-column match.")

    if check_non_builtin_types:
        for v in dictionary.values():
            if (
                not v.__class__.__module__ == "builtins" and
                (allowed_non_builtin_types and not isinstance(v, allowed_non_builtin_types))
            ):
                raise ValueError(f"Found non-allowed non-builtin type object in dictionary.")

    n_rows = df.shape[0]

    # our strategy is to wrap the generic object in a list and expand it to have 
    # the same n_rows of the dataframe. In this way pandas will not consider the 
    # nature of the objects inside the list.
    for k, v in dictionary.items():
        if convert_bool_to_str and isinstance(v, bool):
            v = str(v)
        if convert_none_to_str and v is None:
            v = str(v)
        df[k] = pd.Series([v] * n_rows, dtype="object") \
            if k in force_object_datatype \
            else [v] * n_rows

    return df



def ensure_or_create(obj: Any, constructor: Callable[[], Any]) -> Any:
    """
    Return `obj` if it evaluates to True, 
    otherwise create and return a new instance by calling `constructor`.

    This is useful for safely initializing optional arguments, e.g.:
        my_dict = ensure_or_create(existing_dict, dict)

    Parameters:
        obj (Any):
            The object to check.
        constructor (Callable[[], Any]):
            A zero-argument callable used to create a new object if `obj` is falsy.

    Returns:
        Any: The original object if truthy, otherwise a newly constructed one.
    """
    return obj if obj else constructor()



def enlist(obj: Any, none_as_is: bool = True) -> list|None:
    '''
    Set the obj in a list if not already a list.

    Parameters:
        obj (Any): Object to enlist.
        none_as_is (bool, optional): Whether to leave None as is.
    
    Returns:
        list|None: A list or None object.
    '''
    if none_as_is:
        return obj if isinstance(obj, list) or (obj is None and none_as_is) else [obj]
    else:
        return obj if isinstance(obj, list) else [obj]
    


def strip_level_from_columns(df: pd.DataFrame, level: int | str | list[int] | list[str]) -> pd.DataFrame:
    '''
    Strips level from the columns MultiIndex.
    
    Parameters:
        df (pd.DataFrame): DataFrame with a MultiIndex columns.
        level (int | str | list[int] | list[str]): Level/s to remove.
    
    Returns:
        pd.DataFrame:
        A copy of df with the new columns index.
    '''
    df_stripped = df.copy()
    df_stripped.columns = df.columns.droplevel(level)
    return df_stripped



def select_level_from_columns(df: pd.DataFrame, level: int | str) -> pd.DataFrame:
    '''
    Set a specific level from a columns MultiIndex as the new Index.

    Parameters:
        df (pd.DataFrame): DataFrame with a MultiIndex columns.
        level (int | str): Level to use as the new Index.
    
    Returns:
        pd.DataFrame:
        A copy of df with the new columns index.
    '''
    df_new = df.copy()
    df_new.columns = df.columns.get_level_values(level)
    return df_new



def asdict_shallow(obj_dataclass) -> dict:
    '''
    The standard 'asdict' utility provided by the dataclass module
    acts by default in a recursive way. This utility allows implements
    a shallow version.
    '''
    if not is_dataclass(obj_dataclass):
        raise ValueError("obj_dataclass must be a dataclass instance.")
    return {field.name: getattr(obj_dataclass, field.name) for field in fields(obj_dataclass)}



def subset_1d(obj: YType, idx: None | np.ndarray) -> YType:
    '''
    Subset a pandas series or 1D numpy array with an array of indices.
    If idx is None then the original obj is returned.
    '''
    if idx is None:
        return obj

    if isinstance(obj, pd.Series):
        return obj.iloc[idx]

    return obj[idx]



def subset_2d(obj: XType, idx_rows: None | np.ndarray, idx_cols: None | np.ndarray) -> XType:
    '''
    Subset a pandas DataFrame or 2D numpy array by row and column indices.
    The sets of indices can be None meaning no subset over that dimension.
    '''
    idx_rows = slice(None) if idx_rows is None else idx_rows
    idx_cols = slice(None) if idx_cols is None else idx_cols

    if isinstance(obj, pd.DataFrame):
        return obj.iloc[idx_rows, idx_cols]

    return obj[idx_rows, :][:, idx_cols]



def subset_xy(
    X: XType, 
    y: YType, 
    idx_rows: None | np.ndarray, 
    idx_cols: None | np.ndarray
) -> tuple[XType, YType]:
    '''
    Subset X and y with binary set of indices. 
    The utility assumes that X is 2D and y is 1D.
    '''
    X_sub = subset_2d(X, idx_rows, idx_cols)
    y_sub = subset_1d(y, idx_rows)
    return X_sub, y_sub
