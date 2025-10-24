import numpy as np
import pandas as pd
from preprocessing import DensityFeatureSelector
from preprocessing.utils import get_indexes_to_retain
from preprocessing.utils import get_density_scores



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



def test_exact_selection_is_reproducible():
    '''
    We test that the "exact" strategy is reproducible in presence of ties
    repeating the same selection 20 times. This gives as a high certainty
    that the selection is indeed reproducible.
    '''
    densities = get_mock_data_densities()
    
    last_selected_column = [
        get_indexes_to_retain(densities, n_target=3, strategy="exact")[0][-1]
        for i in range(20)
    ]

    assert len(set(last_selected_column)) == 1, "exact strategy does not ensure reproducibile selection with ties."
    assert last_selected_column[0] == "b1", "exact strategy is not picking the expected tied column."



def test_oversample_selection_is_working():
    '''
    We test whether the oversample strategy is selecting all ties.
    '''
    densities = get_mock_data_densities()
    
    selected_indexes, _ = get_indexes_to_retain(densities, n_target=3, strategy="oversample")
    assert len(selected_indexes) == 4, "oversample strategy is not picking all ties at boundary"

    selected_indexes, _ = get_indexes_to_retain(densities, n_target=1, strategy="oversample")
    assert len(selected_indexes) == 1, "oversample strategy is not working with n_target of 1"



def test_undersample_selection_is_working():
    '''
    We test the different scenarios for the undersample strategy.
    '''
    densities = get_mock_data_densities()
    
    selected_indexes, _ = get_indexes_to_retain(densities, n_target=3, strategy="undersample")
    assert len(selected_indexes) == 1, "undersample strategy is not removing ties when it should"

    selected_indexes, _ = get_indexes_to_retain(densities, n_target=4, strategy="undersample")
    assert len(selected_indexes) == 4, "undersample strategy is removing ties when it should not"

    selected_indexes, _ = get_indexes_to_retain(densities, n_target=1, strategy="undersample")
    assert len(selected_indexes) == 1, "undersample strategy is not working with n_target of 1"



def test_density_feature_selector_is_functional_with_dataframes():
    df = create_mock_data_for_density_selection(as_frame=True)
    dfs = DensityFeatureSelector(n_target_cols=1, strategy="exact")

    try:
        dfs.fit(df)
        dfs.transform(df)
        dfs.fit_transform(df)
    except Exception:
        assert False, "DensityFeatureSelector has problems with dataframes."



def test_density_feature_selector_is_functional_with_arrays():
    array = create_mock_data_for_density_selection(as_frame=False)
    dfs = DensityFeatureSelector(n_target_cols=1, strategy="exact")
    
    try:
        dfs.fit(array)
        dfs.transform(array)
        dfs.fit_transform(array)
    except Exception:
        assert False, "DensityFeatureSelector has problems with numpy arrays."