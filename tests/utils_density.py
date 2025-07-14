import numpy as np
import pandas as pd
from estimators.preprocessing.utils import get_density_scores



def create_mock_data_for_density_selection(as_frame: bool = False) -> pd.DataFrame|np.ndarray:
    '''
    Creates a mock data with known and tied columns density scores.
    '''
    a = [1, 0, 0]
    b = [1, 1, 0]
    c = [1, 1, 1]

    data = pd.DataFrame({
        "a": a,
        "b0": b,
        "b1": b,
        "b2": b,
        "c": c
    })

    if not as_frame:
        data = data.to_numpy()

    return data



def get_mock_data_densities() -> pd.Series:
    '''
    Returns the densities score for the dataframe created by
    "create_mock_data_for_density_selection" utility.
    '''
    df = create_mock_data_for_density_selection(as_frame=True)
    return get_density_scores(df)