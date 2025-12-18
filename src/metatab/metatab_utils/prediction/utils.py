import numpy as np
import pandas as pd
from typing import Iterable, Any
from copy import deepcopy



def wrap_into_list(*objects) -> list[Any] | list[list[Any]]:
    '''Wrap non-list python objects with list.'''
    wrapped = [obj if isinstance(obj, list) else [obj] for obj in objects]
    return wrapped[0] if len(wrapped) == 1 else wrapped



def to_list_of_numpy_arrays(
    iterable: Iterable[np.ndarray | pd.Series],
    copy: bool = False
) -> list[np.ndarray]:
    '''
    Obtain a list of numpy arrays from an iterable of numpy or pandas arrays.
    Parameters:
        iterable (Iterable[np.ndarray|pd.Series]):
            Iterable of pandas series and/or numpy arrays.
        copy (cool, optional):
            Whether to deepcopy the original data in the resulting list.
    '''
    array_list = []
    for el in iterable:
        if isinstance(el, pd.Series):
            array_list.append(el.to_numpy(copy=copy))
        elif isinstance(el, np.ndarray):
            array_to_insert = deepcopy(el) if copy else el
            array_list.append(array_to_insert)
        else:
            raise TypeError("el must be a pandas series or a numpy array.")
    return array_list



def are_same_length(*iterables) -> bool:
    '''Utility to verify if all iterables in the sequence have the same length. Return a bool.'''
    if not iterables:
        return ValueError("No argument passed in input.")
    first_length = len(iterables[0])
    return all([len(iterable) == first_length for iterable in iterables])