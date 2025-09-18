import pandas as pd
from copy import deepcopy
from typing import  Any



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
    dict_keys = dictionary.keys()

    if check_matching_keys_cols:
        for col in df.columns:
            if col in dict_keys:
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
        df[k] = pd.Series([v] * n_rows, dtype="object") if k in force_object_datatype else [v] * n_rows

    return df
