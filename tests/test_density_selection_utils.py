from preprocessing.utils import get_indexes_to_retain
from tests.utils_density import get_mock_data_densities



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
