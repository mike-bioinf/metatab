import pandas as pd
from collections import defaultdict
from estimators.params import HPS_MIXED_TYPES, HPS_COMPLEX_TYPES



class ConfigSearchCV:
    '''Class that holds the globally configurable settings for SearchCV instances'''
    build_df_search = False



def build_hps_dataframe_from_list_of_points(points: list[dict]) -> pd.DataFrame:
    '''
    Build the hyperparameter dataframes from a list of space points.
    The points are expected to have the same set of keys.
    The utility take cares of assigning and respecting the correct value types.
    Returns the HPs dataframe.
    '''
    dict_lists = defaultdict(list)
    to_str_keys = HPS_MIXED_TYPES + HPS_COMPLEX_TYPES

    for point in points:
        for k, v in point.items():
            if k in to_str_keys:
                dict_lists[k].append(str(v))
            else:
                dict_lists[k].append(v)

    return pd.DataFrame(dict(dict_lists))



def aggregate_df_search_at_iteration_level(df_search: pd.DataFrame) -> pd.DataFrame:
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

    del agg_dict["search_iter"]
    df_search = df_search.groupby("search_iter").agg(agg_dict).reset_index()
    
    # we remove the non useful cols on the aggregate result to avoid creating a deepcopy  
    del df_search["fold"]
    del df_search["repeat"]

    return df_search