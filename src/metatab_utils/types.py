import numpy as np
import pandas as pd
from typing import TypeAlias, Union


XType: TypeAlias = Union[
    pd.DataFrame,
    np.ndarray
]

YType: TypeAlias = Union[
    pd.Series,
    np.ndarray
]