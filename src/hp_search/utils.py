import pandas as pd
from collections import defaultdict
from estimators.params import HPS_MIXED_TYPES, HPS_COMPLEX_TYPES



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