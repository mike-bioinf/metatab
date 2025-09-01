import pandas as pd
from typing import Literal, Any



def get_indexes_to_retain(
    densities: pd.Series, 
    n_target: int, 
    strategy: Literal["exact", "oversample", "undersample"]
) -> tuple[list[Any], float]:
    '''
    Get the list of columns indexes to retain to reach the target number of elements.
    The selection is guided by the density scores, i.e. only the n_target 
    most dense columns are kept.
    
    A number of indexes different than n_target can be returned depending
    on the strategy.

    Note: A void list can be returned in some cases, for example
    when n_target is 0 or with the 'undersample' strategy.

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
        tuple:
        The indexes to keep as a list. The list can be void.
        The minimum density score that is kept. It is -1 if no index is kept.
    '''
    if n_target < 0:
        raise ValueError("n_target must be in [0, inf].")
    
    if n_target == 0:
        # we use -1 to indicate that no index is kept.
        # the minimum density score is not determinable in this case.
        return [], -1
    
    sorted_densities = densities.sort_values(ascending=False, kind="stable")

    if densities.size <= n_target:
        return densities.index.to_list(), sorted_densities[-1]
    
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
        right_densities = sorted_densities[n_target:]
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



def get_density_scores(df: pd.DataFrame, axis: int = 0, do_check: bool = False) -> pd.Series:
    '''
    Get the density fraction along an axis.
    The "do_check" flag control whether to check the assumptions 
    made about the dataframe. This is useful to avoid repeating 
    the same checks when using other utilities like the sklearn ones.
    '''
    if do_check: _check_df_assumptions(df, axis)
    return df.apply(lambda x: (x != 0).mean(), axis=axis)



def _check_df_assumptions(df: pd.DataFrame) -> None:
    '''
    Checks the assumptions of the dataframes that we make.
    In detail we expect non-empty, full numeric dataframes,
    without missing values.
    Note: the numeric nature of values is evaluated with the
    `select_dtypes("number")` call.
    '''
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame object.")

    if df.empty:
        raise ValueError("df is empty.")    

    if df.select_dtypes("number").shape[1] != df.shape[1]:
        raise ValueError("There are non numeric elements in df.")
    
    if df.isna().any():
        raise ValueError("There are missing values in df.")
    