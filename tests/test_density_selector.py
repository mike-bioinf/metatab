from fit.preprocessing import DensityFeatureSelector
from tests.utils_tests import create_mock_data_for_density_selection



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
