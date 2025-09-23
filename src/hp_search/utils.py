from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd




class ConfigSearchCV:
    '''Class that holds the globally configurable settings for SearchCV instances'''
    build_df_search = False



def aggregate_df_search_at_iteration_level(
    df_search: pd.DataFrame, 
    remove_groupby_column: bool = False
) -> pd.DataFrame:
    '''
    Abstract the logic to aggregate the "df_search" dataframe at search iteration level,
    computing the mean of cv inner losses for each iteration.
    Note that the function does not control if the dataframe in input is a valid df_search dataframe.
    Returns the aggragated dataframe.
    '''
    agg_dict = {}
    for col in df_search.columns:
        agg_func = "mean" if col == "loss" else "first"
        agg_dict[col] = agg_func

    search_col = "search_iter"
    del agg_dict[search_col]
    df_search = df_search.groupby(search_col).agg(agg_dict).reset_index()

    if remove_groupby_column:
        del df_search[search_col]
    
    # we remove the non useful cols on the aggregate result to avoid creating a deepcopy  
    del df_search["fold"]
    del df_search["repeat"]

    return df_search