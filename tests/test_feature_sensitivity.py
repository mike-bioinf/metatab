import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.datasets import make_classification
from estimators.params import TuningParams
from hp_search.point_corrector import PointCorrector
from metalearning.metadata_generator import MetadataGenerator
from metalearning.sampler import HyperoptRandomSampler
from metalearning.metafeatures import CustomMFE
from metalearning.load import query_surrogate_framework
from metalearning.feature_sensitivity import _permute_block, compute_feature_sensitivity_map
from pymfe.mfe import MFE



@pytest.fixture(scope="module")
def create_multi_index_dataframe() -> pd.DataFrame:
    n_features = 20
    X, y = make_classification(n_samples=10, n_features=n_features, random_state=0)
    outer_level = np.repeat(["A", "B"], n_features/2)
    inner_level = [f"col_{i}" for i in range(n_features)]
    columns = pd.MultiIndex.from_arrays([outer_level, inner_level], names=["group", "feature"])
    X = pd.DataFrame(X, columns=columns)
    y = pd.Series(y)
    return X, y



def test_that_permute_block_is_working_as_expected(create_multi_index_dataframe):
    X, _ = create_multi_index_dataframe
    X = deepcopy(X)

    a_mask = X.columns.get_level_values("group") == "A"
    rng = np.random.default_rng(0)
    index_permutation = rng.permutation(X.shape[0])
    
    X_permuted = _permute_block(X, index_permutation, "group", "A")
    orig_block = X.loc[:, a_mask].to_numpy()
    perm_block = X_permuted.loc[:, a_mask].to_numpy()

    assert (X.columns == X_permuted.columns).all(), "permute_block changes columns order"
    assert np.allclose(orig_block[index_permutation], perm_block), "permute_block permutes rows incorrectly"
    assert np.allclose(X.loc[:, ~a_mask].to_numpy(), X_permuted.loc[:, ~a_mask].to_numpy()), "permute_block permutes non-target columns"




def test_that_compute_feature_sensitivity_map_works(create_multi_index_dataframe):
    X, y = create_multi_index_dataframe
    surrogate_pipeline = query_surrogate_framework("lgbm")
    
    generator = MetadataGenerator(
        sampler=HyperoptRandomSampler(),
        point_corrector=PointCorrector(),
        mfe=CustomMFE()
    )

    metadata, _ = generator.fit(X, y, TuningParams.LGMB_C0, 0).generate(
        n_points=5,
        mfe_extract_kwargs={"add_features": {"preprocessing": "base"}},
        set_metagroups_in_index=True
    )

    map = compute_feature_sensitivity_map(
        model=surrogate_pipeline,
        X=metadata,
        column_index_level="group",
        n_permutations=2
    )

    fe = MFE()
    groups = list(fe.valid_groups()) + ["hps"]
    
    # remove "landmarking" group which is suppressed by our mfe
    groups = [g for g in groups if g != "landmarking"]

    for group in groups:
        if group not in map.keys():
            raise KeyError(f"Metagroup '{group}' is not present in the feature sensitivity map.")
