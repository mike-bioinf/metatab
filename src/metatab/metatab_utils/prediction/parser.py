import io
import base64
import numpy as np
from typing import Any



def safe_ndarray_to_str(value: Any):
    '''ndarray_to_str function with conditional check on input type.'''
    return ndarray_to_str(value) if isinstance(value, np.ndarray) else value


def ndarray_to_str(a: np.ndarray) -> str:
    '''
    Convert the nunmpy array to a string representation with full metadata (shape, dtype, ...).
    Steps: array -> binary -> base64 -> str
    '''
    buffer = io.BytesIO()
    # we not allow pickle since we work with non object-dtyped arrays
    np.save(buffer, a, allow_pickle=False)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def safe_str_to_ndarray(value: Any):
    '''
    str_to_ndarray function with conditional check on str.
    We check that the value is str before transforming it
    since in our columns we can have pd.NA and floats other than
    str representations of numpy arrays, but not "native" strings.
    '''
    return str_to_ndarray(value) if isinstance(value, str) else value


def str_to_ndarray(b64_str: str) -> np.ndarray:
    '''
    Decode the string representation back to the numpy array.
    Steps: str -> base64 -> binary -> array
    '''
    buffer = io.BytesIO(base64.b64decode(b64_str.encode('utf-8')))
    # setting high header size with disabled pickle for better portability
    return np.load(buffer, allow_pickle=False, max_header_size=10_000_000)
