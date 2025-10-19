import pytest
import pandas as pd
from sklearn.datasets import make_classification
from metalearning.metafeatures import CustomMFE
from metalearning.metafeatures import MAP_SPARSE_METAFEATURES


@pytest.fixture(scope="module")
def create_xy_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    n_features = 40
    X, y = make_classification(n_samples=30, n_features=n_features, n_classes=2, random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = pd.Series([f"col_{i}" for i in range(n_features)])
    return X, y


def test_that_custom_mfe_handle_sparse_specifications(create_xy_classification_data):
    X, y = create_xy_classification_data
    custom_mfe = CustomMFE(seed=0)
    custom_mfe.fit(X, y)
    dict_mtf = custom_mfe.extract()


def test_that_custom_mfe_raise_error_with_rescaling_option(create_xy_classification_data):
    X, y = create_xy_classification_data
    custom_mfe = CustomMFE(seed=0)
    with pytest.raises(ValueError, match="Is not possible to compute additional_sparse metafeature on transformed data."):
        custom_mfe.fit(X, y, rescale="standard")


def test_that_custom_mfe_does_not_raise_rescaling_error_with_no_sparse_specification(create_xy_classification_data):
    X, y = create_xy_classification_data
    custom_mfe = CustomMFE(groups="statistical", seed=0)
    custom_mfe.fit(X, y, rescale="standard")


def test_that_custom_mfe_compute_the_sparse_metafeatures(create_xy_classification_data):
    X, y = create_xy_classification_data
    sparse_metafeatures = CustomMFE(groups="additional_sparse", seed=0).fit(X, y).extract()
    assert len(sparse_metafeatures) == len(MAP_SPARSE_METAFEATURES)
    for smtf in MAP_SPARSE_METAFEATURES.keys():
        if smtf not in sparse_metafeatures.keys():
            raise ValueError(
                f"The customMFE does not compute the following sparse metafeature when it should: {smtf}"
            )


def test_that_custom_mfe_add_external_features(create_xy_classification_data):
    X, y = create_xy_classification_data
    metafeatures = CustomMFE(groups="additional_sparse", seed=0).fit(X, y).extract(add_features={"additional": 22})
    try:
        metafeatures["additional"]
    except Exception:
        assert False, "The customMFE does not add the additional features correctly."


def test_that_custom_mfe_raise_error_for_conflicting_added_features(create_xy_classification_data):
    X, y = create_xy_classification_data
    sparse_metafeature = list(MAP_SPARSE_METAFEATURES.keys())[0]
    with pytest.raises(KeyError):
        metafeatures = CustomMFE(groups="additional_sparse", seed=0).fit(X, y).extract(
            add_features={sparse_metafeature: 22}
        )


def test_that_custom_mfe_compute_the_selected_sparse_metafeatures_only(create_xy_classification_data):
    X, y = create_xy_classification_data
    custom_mfe = CustomMFE(groups="additional_sparse", features="fraction_full_zero_columns", seed=0)
    custom_mfe.fit(X, y)
    dict_mtf = custom_mfe.extract()
    mtfs = list(dict_mtf.keys())
    assert len(mtfs) == 1, f"Expected mfe to extract only one metafeatures, but insted {len(mtfs)} are computed."
    assert mtfs[0] == "fraction_full_zero_columns", "Expected mfe to extract the 'fraction_full_zero_columns' metafeature."