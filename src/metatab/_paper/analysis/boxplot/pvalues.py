import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any, Literal
from functools import reduce
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, false_discovery_control
from metatab._paper.analysis.boxplot.types import simple_pair, complex_pair, simple_pairs, complex_pairs
from metatab._paper.analysis.boxplot.pairs import yield_list_same_category_indices, select_pairs



def correct_pvalues(
    pvalues: np.ndarray, 
    pairs: simple_pairs | complex_pairs, 
    strategy: None | Literal["inner_x", "all"] | simple_pairs | complex_pairs, 
    method: Literal["bh", "hb"] = "bh"
) -> list[np.ndarray, np.ndarray]:
    '''
    Correct pvalues using different correction strategies and methods.
    Returns the array with the corrected pvalues.
    Note the resulting arrays can contains also non-corrected pvalues.
    Here "corrected" means post correction. 

    Parameters:
        pvalues: Array of p-values to potentially correct.
        pairs: The pairs corresponding to the pvalues array.
        strategy: How to perform multiple hypothesis correction:
        - None, no correction is done
        - "inner_x", correction is done separately for each category
        - "all", correction is done for all comparisons
        - explicit pairs, correction is done only on specified pairs.
        method: Correction method, either "bh" (Benjamini-Hochberg) or "by" (Benjamini-Yekutieli)
    
    Returns: 
        list[np.ndarray,np.ndarray,np.ndarray]:
        A list of threee arrays:
        1. The corrected pvalues
        2. The correction flags
        3. The correction groups
    '''
    if strategy is None:
        return [pvalues, np.array([False] * len(pvalues)), np.array([np.nan] *  len(pvalues))]

    if strategy == "all":
        post_correction_pvalues = safe_false_discovery_control(pvalues, method)
        return [
            post_correction_pvalues, 
            np.array([True] * len(post_correction_pvalues)),
            np.array([0] * len(post_correction_pvalues))
        ]

    elif strategy == "inner_x":
        post_correction_pvalues = deepcopy(pvalues)
        array_flag_correction = np.array([False] * len(post_correction_pvalues))
        array_group_correction = np.array([np.nan] * len(post_correction_pvalues))

        for idx_group, indices_pairs_category in enumerate(yield_list_same_category_indices(pairs)):
            pvalues_to_correct = np.array([pvalues[i] for i in indices_pairs_category])
            pvalues_adjusted = safe_false_discovery_control(pvalues_to_correct, method)
            # put back the corrected pvalue and flag at their place
            for idx, val in zip(indices_pairs_category, pvalues_adjusted):
                post_correction_pvalues[idx] = val
                array_flag_correction[idx] = True
                array_group_correction[idx] = idx_group
    
    else:
        post_correction_pvalues = deepcopy(pvalues)
        array_flag_correction = np.array([False] * len(post_correction_pvalues))
        array_group_correction = np.array([np.nan] * len(post_correction_pvalues))

        indices_pairs_to_correct = select_pairs(pairs=pairs, to_select=strategy, to_return="indices")
        pvalues_to_correct = np.array([pvalues[i] for i in indices_pairs_to_correct])
        pvalues_adjusted = safe_false_discovery_control(pvalues_to_correct, method)
        
        # put back the corrected pvalue and flag at their place
        for idx, val in zip(indices_pairs_to_correct, pvalues_adjusted):
            post_correction_pvalues[idx] = val
            array_flag_correction[idx] = True
            array_group_correction[idx] = 0

    return [post_correction_pvalues, array_flag_correction, array_group_correction]



def safe_false_discovery_control(pvalues: np.ndarray, method: Literal["bh", "by"] = "bh") -> np.ndarray:
    '''
    Version of 'false_discovery_control' utility that excludes nan from correction.
    The nan are excluded in the correction process but then returned in the post-corrected pvalues.
    Returns the post-correction pvalues in a numpy array.
    '''
    post_correction_pvalues = deepcopy(pvalues)
    finite_mask = np.isfinite(post_correction_pvalues)
    
    post_correction_pvalues[finite_mask] = false_discovery_control(
        post_correction_pvalues[finite_mask], 
        method=method
    )

    return post_correction_pvalues



def execute_test(
    a: np.ndarray, 
    b: np.ndarray, 
    test: Literal["Mann-Whitney", "t-test", "Wilcoxon"], 
    **test_params
) -> float:
    '''Execute a single test on a and b arrays. Returns the pvalue.'''
    if test == "t-test":
        test_func = ttest_ind
    elif test == "Mann-Whitney":
        test_func = mannwhitneyu
    elif test == "Wilcoxon":
        test_func = wilcoxon
    else:
        raise ValueError(f"test not supported: {test}")

    test_object = test_func(a, b, **test_params)
    return test_object.pvalue



def get_stat_arrays_from_pairs(
    df: pd.DataFrame, 
    pairs: simple_pairs | complex_pairs, 
    pair_columns: list[str], 
    stat_column: str,
    paired_column: str | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Get the statistical column arrays for input pairs as a list of tuples.
    Each tuple in the list results from the equivalent position-wise pair in pairs.
    '''
    return [
        get_stat_arrays_from_pair(df, pair, pair_columns, stat_column, paired_column) 
        for pair in pairs
    ]
 


def get_stat_arrays_from_pair(
    df: pd.DataFrame, 
    pair: simple_pair | complex_pair, 
    pair_columns: list[str],
    stat_column: str,
    paired_column: str | None
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Get the statistical column values as numpy arrays for a pair.
    Returns a binary tuple of arrays, one for each element of the pair.
    '''
    stat_arrays = []
    paired_info = []

    for element in pair:
        element = element if isinstance(element, tuple) else (element, )  
        df_pair = filter(df, columns=pair_columns, values=element)

        if df_pair.empty:
             raise ValueError(
                f"There are no values in data for the element '{element}' of the pair '{pair}'."
            )

        stat_array = df_pair[stat_column].to_numpy()    
        stat_arrays.append(stat_array)
        
        if paired_column:
            paired_info.append(df_pair[paired_column])

    if paired_column:  
        check_paired_info(paired_info, pair)
        stat_arrays = order_stat_arrays_based_on_paired_info(stat_arrays, paired_info)
    
    return tuple(stat_arrays)



def filter(df: pd.DataFrame, columns: list[str | tuple[str]], values: tuple[Any]) -> pd.DataFrame:
    '''
    Filter the dataframe based on equality of columns value.
    Allows to filter only those rows for which column == value is True for every (column, value) couple.
    The column, value coupling is order inferred from the 'columns' and 'values' iterables.
    Returns the filtered dataframe.
    '''
    condition_series = [df[column] == values[i] for i, column in enumerate(columns)]
    combined_condition = reduce(lambda s1, s2: s1 & s2, condition_series)
    return df.loc[combined_condition, :]



def check_paired_info(paired_series: list[pd.Series], pair: simple_pair | complex_pair) -> None:
    '''
    Check that the paired info extracted from the two elements of the pair is compatible.
    In detail it checks that the two id series have:
    - same length,
    - same unique values,
    - no duplicated within them.
    '''
    s1, s2 = paired_series

    # check duplicates
    if s1.duplicated().any():
        raise ValueError(f"Found duplicated ids for the first element of {pair}.")
    if s2.duplicated().any():
        raise ValueError(f"Found duplicated ids for the second element of {pair}.")
    
    # check same number of categories
    if s1.size != s2.size:
        raise ValueError(f"Differen number of ids for the pair {pair}.")

    # check same categories
    unique_first = set(s1.unique())
    unique_second = set(s2.unique())

    if unique_first != unique_second:
        raise ValueError(
            f"The paired ids differ for {pair} pair."
        )



def order_stat_arrays_based_on_paired_info(
    stat_arrays: list[np.ndarray], 
    paired_info: list[pd.Series]
) -> list[np.ndarray]:
    '''Orders and returns the statistical arrays based on the paired ids'''
    arr1, arr2 = stat_arrays
    s1, s2 = paired_info

    df1 = pd.DataFrame({"key": s1, "arr1": arr1})
    df2 = pd.DataFrame({"key": s2, "arr2": arr2})

    merged = pd.merge(df1, df2, on="key")
    return [merged["arr1"].to_numpy(), merged["arr2"].to_numpy()]