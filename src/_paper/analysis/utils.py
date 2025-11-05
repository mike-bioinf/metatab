import re
import pandas as pd
import warnings
from typing import Any, Callable, Literal
from estimators.types import ALL_ESTIMATOR_TYPE
from estimators.constants import GBDT_ESTIMATORS



def check_presence_cols(df: pd.DataFrame, cols: str | list[str]) -> None:
    '''
    Check the presence of multiple columns in a dataframe.
    The utility assumes that the columns are strings.
    '''
    checked_cols = []
    for col in enlist(cols):
        if col not in checked_cols:
            check_presence_col(df, col)
            checked_cols.append(col)



def check_presence_col(df: pd.DataFrame, col: str) -> None:
    '''
    Check the presence of a single column in df.
    The utility assumes that the df columns are strings.
    '''
    if col not in df.columns:
        raise ValueError(f"'{col}' column is not found in df.")



def check_numeric_column(df: pd.DataFrame, col: str):
    '''Check the numeric nature of the column'''
    current_dtype = df.dtypes[col]
    if not pd.api.types.is_numeric_dtype(current_dtype):
        raise TypeError(
            f"column '{col}' is expected to be numeric, instead it is of type '{str(current_dtype)}'."
        )



def append_if_not_none(l: list, obj: Any) -> list:
    '''
    Append the object to the list if not None.
    Returns the list.
    '''
    if not isinstance(l, list):
        raise TypeError("l msut be a list object.")
    
    if obj is not None:
        l.append(obj)

    return l



def enlist(x: Any | list, none_as_is = False) -> list:
    '''
    Put into a list x if not already a list.
    If x is None one can decide to not enlist it.
    '''
    if isinstance(x, list):
        return x
    elif x is None and none_as_is:
        return None
    else:
        return [x]



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



def safe_update_dict(
    dict: dict, 
    updating_dict: dict, 
    raise_on_existing_key: Literal["error", "warning"] = "error"
) -> dict:
    '''
    Utility to update a dict with another one raising 
    a condition when overwriting already existing keys.

    Parameters:
        dict (dict): The dict to update.
        updating_dict (dict): The dict updating the first one.
        raise_on_existing_key (Literal["error", "warning"], optional):
            Whether to raise an error or warning when an existing key is to be override.
    
    Returns:
        dict: The updated dict.
    '''    
    for new_key, value in updating_dict.items():
        if new_key in dict.keys():
            if raise_on_existing_key == "warning":
                warnings.warn(f"The key '{new_key}' is override with a new value.")
            else:
                raise ValueError(f"The key '{new_key}' already exists.")
        dict[new_key] = value
        
    return dict



def filter_hpo(
    df_hpo: pd.DataFrame,
    estimators: None | str | list[str] = None,
    spaces: None | str | list[str] = None,
    datasets: None | str | list[str] = None,
    estimator_column = "estimator",
    space_column = "configuration",
    dataset_column = "dataset"
) -> pd.DataFrame:
    '''
    Utility to filter hpo dataframes.
    Filter the rows that satisfy all values (AND concatenation) 
    specified in the defined estimators, spaces and datasets parameters.
    If these parameters are set to None then no filtering is done on them.
    '''
    mask = pd.Series(True, index=df_hpo.index)
    
    if estimators is not None:
        check_presence_col(df_hpo, estimator_column)
        mask &= df_hpo[estimator_column].isin(enlist(estimators))
    if spaces is not None:
        check_presence_col(df_hpo, space_column)
        mask &= df_hpo[space_column].isin(enlist(spaces))
    if datasets is not None:
        check_presence_col(df_hpo, dataset_column)
        mask &= df_hpo[dataset_column].isin(enlist(datasets))

    return df_hpo.loc[mask, :]



def is_early_stopped_estimator(estimator: ALL_ESTIMATOR_TYPE, invert: bool = False) -> bool:
    '''
    Utility to infer whether the estimator is early stopped or not based on its "str type".

    Parameters:
        estimator (ALL_ESTIMATOR_TYPE): Estimator "str type".
        invert (bool, optional): Whether to invert the check.

    Returns:
        bool.
    '''
    is_es = True if re.match("es_", estimator) else False
    return not is_es if invert else is_es



def is_gbdt_estimator(estimator: ALL_ESTIMATOR_TYPE, invert: bool = False) -> bool:
    '''
    Utility to infer whether the estimator belongs to the gbdt family based on its "str type"

    Parameters:
        estimator (ALL_ESTIMATOR_TYPE): Estimator "str type".
        invert (bool, optional): Whether to invert the check.
    
    Returns:
        bool
    '''
    is_gbdt = estimator in GBDT_ESTIMATORS
    return is_gbdt if not invert else not is_gbdt



def refine_tune_info(
    tune_algo_series: pd.Series,
    distinguish_tpe_from_random: bool = False
) -> pd.Series:
    '''
    Refine the tune info using the "tune_algo" column.

    Parameters:
        tune_algo_series (pd.Series): Tune algo series.

        distinguish_tpe_from_random (bool, optional):
            Whether distinguish the refined value of random and tpe algo.
            If False then "HPO" is the refined value for both of them.
            If True then "tpe" maps to "MHPO" and "random" to "RHPO"
    
    Returns:
        pd.Series: The refined values
    '''
    refined_values = []

    for value in tune_algo_series:
        if pd.isna(value):
            refined_values.append("DF")
        elif value == "random":
            rvalue = "RHPO" if distinguish_tpe_from_random else "HPO"
            refined_values.append(rvalue)
        elif value == "tpe":
            rvalue = "THPO" if distinguish_tpe_from_random else "HPO"
            refined_values.append(rvalue)
        elif value == "meta":
            refined_values.append("MHPO")
        else:
            raise ValueError(f"Unrecognized value encountered: {value}")

    return pd.Series(refined_values)



def simple_refine_tune_info(tune_series: pd.Series, estimator_series: pd.Series) -> pd.Series:
    '''
    Refine the tune info using the "tune" boolean info.

    Parameters:
        tune_series (pd.Series): Series with the boolean tune info.
        estimator_series (pd.Series): Series with the estimator str info.

    Returns:
        pd.Series: The refined 'estimator_tune' series.
    '''
    if len(tune_series) != len(estimator_series):
        raise ValueError("'tune_series' and 'estimator_series' have different lengths")

    res = []
    for t, e in zip(tune_series, estimator_series):
        v = e + "_HPO" if t else e + "_DF"
        res.append(v)

    return pd.Series(res, index=tune_series.index)



def create_early_stopped_estimator_column(
    df: pd.DataFrame, 
    estimator_column: str = "estimator", 
    new_column: str = "is_early_stopped",
    check_collision_new_column: bool = True,
    copy: bool = False
) -> pd.DataFrame:
    '''
    Create a boolean column indicating whether the estimator is early stopped.
    The condition is evaluated using the "is_early_stopped_estimator" utility.

    Parameters:
        df (pd.DataFrame): Dataframe.
        
        estimator_column (str, optional): 
            Name of the column with the estimator info on which the check is based.
        
        new_column (str, optional): Name of the column created.

        check_collision_new_column (bool, optional):
            Whether to check if the the `new_column` already exists in df.
            If this is True an error is raised. 
            If False the column will be replaced by the new created one.

        copy (bool, optional):
            Whether to act on a deepcopy of the original dataframe.

    Returns:
        pd.DataFrame: The dataframe with the added column.
    '''
    check_presence_col(df, estimator_column)
    
    if check_collision_new_column and new_column in df.columns:
        raise ValueError(f"{new_column} already exists in df.")
    
    if copy: 
        df = df.copy()
    
    df[new_column] = df[estimator_column].apply(is_early_stopped_estimator)
    return df



def create_gbdt_estimator_column(
    df: pd.DataFrame, 
    estimator_column: str = "estimator", 
    new_column: str = "is_gbdt",
    check_collision_new_column: bool = True,
    copy: bool = False
):
    '''
    Create a boolean column indicating whether the estimator belongs to the GBDT family.
    The condition is evaluated using the "is_gbdt_estimator" utility.

    Parameters:
        df (pd.DataFrame): Dataframe.
        
        estimator_column (str, optional): 
            Name of the column with the estimator info on which the check is based.
        
        new_column (str, optional): Name of the column created.

        check_collision_new_column (bool, optional):
            Whether to check if the the `new_column` already exists in df.
            If this is True an error is raised. 
            If False the column will be replaced by the new created one.

        copy (bool, optional):
            Whether to act on a deepcopy of the original dataframe.

    Returns:
        pd.DataFrame: The dataframe with the added column.
    '''
    check_presence_col(df, estimator_column)
    
    if check_collision_new_column and new_column in df.columns:
        raise ValueError(f"{new_column} already exists in df.")
    
    if copy: 
        df = df.copy()

    df[new_column] = df[estimator_column].apply(is_gbdt_estimator)
    return df
