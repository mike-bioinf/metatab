import pandas as pd
from typing import Any



def check_presence_cols(df: pd.DataFrame, cols: str | list[str]) -> None:
    '''
    Check the presence of multiple columns in df.
    The utility assumes that the df columns are strings.
    '''
    for col in enlist(cols):
        check_presence_col(df, col)



def check_presence_col(df: pd.DataFrame, col: str) -> None:
    '''
    Check the presence of a single column in df.
    The utility assumes that the df columns are strings.
    '''
    if col not in df.columns:
        raise ValueError(f"'{col}' column is not found in df.")



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