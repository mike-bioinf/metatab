import pandas as pd
from typing import Literal, Any, Callable


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


def check_presence_cols(df: pd.DataFrame, cols: str | list[str]) -> None:
    '''
    Check the presence of multiple columns in a dataframe.
    The utility assumes that the columns are strings.
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


def _compute_normalized_score(s: pd.Series) -> pd.Series:
    '''Computes the normalized scores following TabRepo'''
    median_s = s.median()
    return ((s - median_s) / (s.max() - median_s)).apply(lambda v: max(0, v))


def compute_aggregate_statistics(
    d_wide: pd.DataFrame,
    skip_secondary_stats: bool = True
) -> pd.DataFrame:
    '''
    Compute aggregate statistics on a wide dataframe (dataset x classifier).
    The function expects datasets in index and classifiers in columns.

    In detail computes:
        -average_rank --> average of per-row ranks across rows.
        -std_rank --> std in ranks across rows.
        -wins --> Absolute number of times each classifier is the best per row across rows.
        -wins_ratio --> express #wins over the total number of rows.
        -average_regret --> average regret (distance from the best per row) across rows.
        -std_regret --> std of regret across rows.
        -improvability --> improvability metric defined following TabArena.
        -normalized_score --> score defined in TabRepo.
        -quartiles --> median, q25 and q75 values.

    Notes: 
    1) Ranking and winning metrics use the average method to resolve ties.
    This means that wins do not include draws. 
    2) Ranking, winning, improvability, regret and normalized scores 
    metrics are computed across classifiers and then averaged across datasets.
    Quartiles are instead compute directly across datasets by classifier.
    
    Parameters:
        d_wide (pd.DataFrame): 
            Wide DataFrame with datasets in index and classifier in columns.
        skip_secondary_stats (bool, optional):
            Whether to skip regret and normalized score statistics computation.

    Returns:
        pd.DataFrame: DataFrame with all the aggregate statistics. 
    '''
    d_rank = d_wide.rank(axis=1, ascending=False, method="average") # ties are resolved as averages
    d_improvabity = d_wide.apply(lambda row: ((row.max() - row) / (1 - row) * 100), axis=1)
    wins = (d_rank == 1).sum(axis=0)

    stats = pd.DataFrame({
        # Rank statistics
        "average_rank": d_rank.mean(axis=0),
        "std_rank": d_rank.std(axis=0),
        # Win statistics
        "wins": wins,
        "wins_ratio": wins / d_wide.shape[0],
        # Improvability
        "average_improvability": d_improvabity.mean(axis=0),
        "std_improvability": d_improvabity.std(axis=0),
        # Raw score statistics
        "median_score": d_wide.median(axis=0),
        "score_q25": d_wide.quantile(0.25, axis=0),
        "score_q75": d_wide.quantile(0.75, axis=0),
    })

    if not skip_secondary_stats:
        d_regret = d_wide.apply(lambda row: row.max() - row, axis=1)
        stats_secondary = pd.DataFrame({
            # Regret statistics
            "average_regret": d_regret.mean(axis=0),
            "std_regret": d_regret.std(axis=0),
            # Normalized scores
            "average_normalized_score": d_wide.apply(lambda row: _compute_normalized_score(row), axis=1).mean(axis=0),
        })
        stats = pd.concat([stats, stats_secondary], axis=1)

    return stats