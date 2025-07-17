import numpy as np
import pandas as pd
from typing import Iterable



def wrap_into_list(*objects) -> list | list[list]:
    '''Wrap non-list python objects with list.'''
    wrapped = [obj if isinstance(obj, list) else [obj] for obj in objects]
    return wrapped[0] if len(wrapped) == 1 else wrapped



def to_numpy_iterable(iterable: Iterable[np.ndarray | pd.Series]) -> list[np.ndarray]:
    '''Create from an iterable of numpy arrays or pandas series a list of numpy arrays.'''
    if not iterable:
        return iterable
    numpy_iterable = []
    for el in iterable:
        numpy_iterable.append(el.to_numpy(copy=True) if isinstance(el, pd.Series) else el)
    return numpy_iterable



def are_same_length(*iterables) -> bool:
    '''Utility to verify if all iterables in the sequence have the same length. Return a bool.'''
    if not iterables:
        return ValueError("No argument passed in input.")
    first_length = len(iterables[0])
    return all([len(iterable) == first_length for iterable in iterables])