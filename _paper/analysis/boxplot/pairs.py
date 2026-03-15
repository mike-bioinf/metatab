import numpy as np
import pandas as pd
from typing import Literal, Iterable, Iterator, Any, Generator
from itertools import combinations, product
from metatab._paper.analysis.boxplot.types import simple_pair, simple_pairs, complex_pair, complex_pairs



def extract_pairs(
    data: pd.DataFrame, 
    column_categories: str, 
    column_levels: None | str = None, 
    pair_type: Literal["inner_x", "all"] = "inner_x"
) -> simple_pairs | complex_pairs:
    '''
    Utility to get the pairs to compare and to annotate.

    Parameters:
        data (pd.DataFrame):
            Dataframe with string column indexes.
        
        column_categories (str): 
            Name of the column on which the x categories are inferred.
        
        column_levels (None | str, optional): 
            Name of the column on which the categories inner-levels are inferred. 
            Defaults to None, in which case the pairs are inferred only on the category column.
        
        pair_type (Literal["inner_x", "all"]):
            Specify how the pairs are inferred:
            - inner_x, the pairs between the inner levels of each x category are returned.
            If no inner levels are given then returns the pairs between all x categories.
            - all, the pairs between all levels of all categories, or between all categories only 
            (if levels are missing) are returned.

    Returns:
        simple_pairs|complex_pairs: The pairs as a list of tuples.
    '''
    pairs = []

    if column_levels is None:
        # pairs categories only
        categories = data[column_categories].unique()
        pairs = list(safe_combinations(categories, 2))

    elif pair_type == "inner_x":
        for cat, group in data.groupby(column_categories):
            # combining (cat, level) tuples intra-category
            cat_levels = list(product((cat,), group[column_levels].unique()))
            # skipping single cat_level in group --> no pair is possible
            if len(cat_levels) > 1:
                pairs.extend(combinations(cat_levels, 2))
                
        if len(pairs) == 0:
            raise ValueError(
                f"All categories have only a single level. No intra-category pairs are possible."
            )

    elif pair_type == "all":
        cat_levels = []
        for cat, group in data.groupby(column_categories):
            cat_levels.extend(product((cat,), group[column_levels].unique()))
        # combining all possibles (cat, level) tuples
        pairs = list(safe_combinations(cat_levels, 2))

    return pairs
        


def safe_combinations(iterable: Iterable, r: int) -> Iterator:
    '''Safe version of combinations utility that raise an error in the input iterable is of length 1'''
    if len(iterable) == 1:
        raise ValueError(f"Cannot create combinations of length {r} from iterable of length 1.")
    return combinations(iterable, r)



def get_unique_categories_from_pairs(pairs: complex_pairs) -> list[Any]:
    return list(set([pair[0][0] for pair in pairs]))



def get_indices_same_category_pairs(pairs: complex_pairs) -> np.ndarray[int]:
    '''Get the indices of the same category pairs as a numpy array of integers.'''
    return np.array([i for i, pair in enumerate(pairs) if pair[0][0] == pair[1][0]], dtype=np.int64)



def get_indices_different_category_pairs(pairs: complex_pairs) -> np.ndarray[int]:
    '''Get the indices of the different category pairs as a numpy array of integers.'''
    return np.array([i for i, pair in enumerate(pairs) if pair[0][0] != pair[1][0]], dtype=np.int64)



def get_same_category_pairs(pairs: complex_pairs) -> complex_pairs:
    '''Select the pairs with the same category value, aka the same first value.'''
    return [pair for pair in pairs if pair[0][0] == pair[1][0]]



def get_different_category_pairs(pairs:complex_pairs) -> complex_pairs:
    '''Select the pairs with different category value, aka different first value.'''
    return [pair for pair in pairs if pair[0][0] != pair[1][0]]



def yield_list_same_category_indices(pairs: complex_pairs) -> Generator[list[int], None, None]:
    '''Yields lists of pairs indices with the same category for both element divided by category'''
    same_category_indexes = get_indices_same_category_pairs(pairs)
    for category in get_unique_categories_from_pairs(pairs):
        yield [
            index 
            for index in same_category_indexes 
            if pairs[index][0][0] == category
        ]



def is_complex_pairs(pairs: simple_pairs | complex_pairs) -> bool:
    '''Test whether the pairs are simple or complex returning a boolean.'''
    return True if isinstance(pairs[0][0], tuple) else False



def are_equal_pairs(pair_0: simple_pair | complex_pair, pair_1: simple_pair | complex_pair) -> bool:
    '''
    Test if two simple or complex pairs are equal.
    The pair element orders has no meaning.
    Returns a bool.
    '''
    el_00, el_01 = pair_0
    el_10, el_11 = pair_1

    return (
        (el_00 == el_10 and el_01 == el_11) or
        (el_00 == el_11 and el_01 == el_10)
    )
    


def select_pairs(
    pairs: simple_pairs | complex_pairs, 
    to_select: simple_pairs | complex_pairs,
    to_return: Literal["pairs", "indices"] = "pairs", 
    inverse: bool = False
) -> simple_pairs | complex_pairs | list[int]:
    '''
    Select specific pairs based on pair equality.
    Returns an empty list if no pair is selected.

    Parameters:
        pairs (simple_pairs | complex_pairs): 
            Pairs on which the selection process is performed.
        
        to_select (simple_pairs | complex_pairs): 
            Pairs to select.
        
        to_return (Literal["pairs", "indices"], optional): 
            Whether to return the pairs or their indices.
        
        inverse (bool, optional): 
            Whether to reverse the selection process. Default to False.

    Returns:
        simple_pairs|complex_pairs|list[int]: The selected pairs/index-pairs.
    '''
    selected = []

    for i, pair in enumerate(pairs):
        object_to_select = pair if to_return == "pair" else i

        for pair_in_selection in to_select:
            is_pair_into_select = are_equal_pairs(pair, pair_in_selection)
            if is_pair_into_select: break
        
        if is_pair_into_select and not inverse:
            selected.append(object_to_select) 
        elif not is_pair_into_select and inverse:
            selected.append(object_to_select)

    return selected