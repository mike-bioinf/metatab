import pytest
import numpy as np
import pandas as pd
from metatab.preprocessing import DensityFeatureSelector



def create_mock_data_for_density_selection(as_frame: bool = False) -> pd.DataFrame | np.ndarray:
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



def test_density_feature_selector_is_functional_with_dataframes():
    df = create_mock_data_for_density_selection(as_frame=True)
    dfs = DensityFeatureSelector(n_target_cols=1, strategy="exact")

    try:
        dfs.fit(df)
        dfs.transform(df)
        dfs.fit_transform(df)
    except Exception:
        assert False, "DensityFeatureSelector has problems with pandas dataframes."



def test_density_feature_selector_is_functional_with_arrays():
    array = create_mock_data_for_density_selection(as_frame=False)
    dfs = DensityFeatureSelector(n_target_cols=1, strategy="exact")
    
    try:
        dfs.fit(array)
        dfs.transform(array)
        dfs.fit_transform(array)
    except Exception:
        assert False, "DensityFeatureSelector has problems with numpy arrays."



def test_density_feature_selector_set_output_api_is_working():
    X = create_mock_data_for_density_selection(as_frame=True)
    density_selector = DensityFeatureSelector(n_target_cols=3, strategy="oversample").set_output(transform="pandas")
    X_trans = density_selector.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame), "DensityFeatureSelector set_output API is not returning dataframes when requested."
    assert X_trans.columns.to_list() == ["b0", "b1", "b2", "c"], "DensityFeatureSelector set_output API is returning the wrong names/order-names"



def test_density_feature_selector_on_empty_mechanism_is_working():
    X = create_mock_data_for_density_selection(as_frame=True)
    density_selector = DensityFeatureSelector(n_target_cols=0, strategy="oversample", on_empty="error").set_output(transform="pandas")

    with pytest.raises(match="Feature selection resulted in an empty feature set."):
        X_trans = density_selector.fit_transform(X)

    density_selector = DensityFeatureSelector(n_target_cols=0, strategy="oversample", on_empty="select_all").set_output(transform="pandas")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.columns.to_list() == X.columns.to_list(), "DensityFeatureSelector 'select_all' option of on_empty mechanism is not working."



def test_exact_selection_is_reproducible():
    '''
    We test that the "exact" strategy is reproducible in presence of ties
    repeating the same selection 20 times. This gives as a high certainty
    that the selection is indeed reproducible.
    '''
    X = create_mock_data_for_density_selection(as_frame=True)
    selected_cols = []

    for _ in range(20):
        density_selector = DensityFeatureSelector(n_target_cols=3, strategy="exact").set_output(transform="pandas")
        X_trans = density_selector.fit_transform(X)
        selected_cols.append(X_trans.columns.to_list())
        
    for i in range(len(selected_cols)):
        for j in range((i+1), len(selected_cols)):
            assert selected_cols[i] == selected_cols[j], "exact strategy does not ensure reproducibile selection with ties."



def test_oversample_selection_is_working():
    '''
    We test whether the oversample strategy is selecting all ties.
    '''
    X = create_mock_data_for_density_selection(as_frame=True)

    density_selector = DensityFeatureSelector(n_target_cols=3, strategy="oversample")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.shape[1] == 4, "oversample strategy is not picking all ties at boundary"

    density_selector = DensityFeatureSelector(n_target_cols=1, strategy="oversample")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.shape[1] == 1, "oversample strategy is not working with n_target of 1"



def test_undersample_selection_is_working():
    '''
    We test the different scenarios for the undersample strategy.
    '''
    X = create_mock_data_for_density_selection(as_frame=True)

    density_selector = DensityFeatureSelector(n_target_cols=3, strategy="undersample")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.shape[1] == 1, "undersample strategy is not removing ties when it should"

    density_selector = DensityFeatureSelector(n_target_cols=4, strategy="undersample")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.shape[1] == 4, "undersample strategy is removing ties when it should not"

    density_selector = DensityFeatureSelector(n_target_cols=1, strategy="undersample")
    X_trans = density_selector.fit_transform(X)
    assert X_trans.shape[1] == 1, "undersample strategy is not working with n_target of 1"