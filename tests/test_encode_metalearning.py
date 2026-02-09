import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from sklearn.pipeline import make_pipeline
from metatab.estimators.utils.types import TunableEstimatorType
from metatab.metalearning.encode.encode import get_encoding_scheme
from metatab.metalearning.encode.transformers import NanToNone, ColToStr, InfToNan
from metatab.preprocessing.types import PreprocessingStrategy



def load_metadata(
    estimator: TunableEstimatorType,
    preprocessing: PreprocessingStrategy = "base"
) -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "data/metadata" / f"{estimator}.txt", sep="\t")
    df["preprocessing"] = preprocessing
    return df



@pytest.mark.skipif(not (Path(__file__).parent / "data/metadata").exists(), reason="Missing Metadata")
def test_that_nan_to_none_transformer_works():
    metadata = load_metadata("tabpfn")
    assert metadata["inference_config__OUTLIER_REMOVAL_STD"].isna().any()
    nan_to_none = NanToNone("inference_config__OUTLIER_REMOVAL_STD").set_output(transform="pandas")
    # test that set_output framework works
    trans_metadata = nan_to_none.fit_transform(metadata)
    assert isinstance(trans_metadata, pd.DataFrame)
    assert (trans_metadata.columns == metadata.columns).all(), "NoneToNone transformer changes column names"
    # test that nan are converted to None
    is_none_list = [el is None for el in trans_metadata["inference_config__OUTLIER_REMOVAL_STD"]]
    is_nan_list = [pd.isna(el) for el in trans_metadata["inference_config__OUTLIER_REMOVAL_STD"] if el is not None]
    assert any(is_none_list), "NanToNone transformer is not converting nan to None"
    assert not any(is_nan_list), "NanToNone transformer is not converting nan to None"



@pytest.mark.skipif(not (Path(__file__).parent / "data/metadata").exists(), reason="Missing Metadata")
def test_that_col_to_str_transformer_works():
    metadata = load_metadata("tabpfn")
    assert pd.api.types.is_numeric_dtype(metadata["z_normalized_loss"].dtype)
    col_to_str = ColToStr("z_normalized_loss").set_output(transform="pandas")
    trans_metadata = col_to_str.fit_transform(metadata)
    # test that set_output framework works
    assert isinstance(trans_metadata, pd.DataFrame)
    assert (metadata.columns == trans_metadata.columns).all(), "ColToStr transformer chnages column names"
    # test that the transformer convert to str
    assert pd.api.types.is_object_dtype(trans_metadata["z_normalized_loss"].dtype)    
    for value in trans_metadata["z_normalized_loss"]:
        assert isinstance(value, str), "ColToStr doesn't cast to str."



@pytest.mark.skipif(not (Path(__file__).parent / "data/metadata").exists(), reason="Missing Metadata")
def test_that_inf_to_nan_transformer_works():
    X = pd.DataFrame([[np.inf, -np.inf], [1, None]], dtype="object")
    transformer = InfToNan().set_output(transform="pandas")
    X_trans = transformer.fit_transform(X)
    assert X_trans.iloc[1, 1] is None, "InfToNan automatically downcast columns and values."
    assert X_trans.iloc[0, :].isna().sum() == 2, "InfToNan does not convert +/- inf values to nan"



## TODO: add "catboost", "extra_trees", "realmlp", "tabm" when you have metadata
@pytest.mark.parametrize("estimator", ["random_forest", "xgb", "lgbm", "tabpfn"])
@pytest.mark.skipif(not (Path(__file__).parent / "data/metadata").exists(), reason="Missing Metadata")
def test_that_estimator_metadata_encoding_scheme_is_correct(estimator):
    enc_pipe = make_pipeline(*get_encoding_scheme(estimator)).set_output(transform="pandas")
    # suppress variance threshold warnings on full na slices
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice.*", category=RuntimeWarning)
        warnings.filterwarnings(action="ignore",message="All-NaN slice encountered", category=RuntimeWarning)
        trans_metadata = enc_pipe.fit_transform(load_metadata(estimator))
    assert isinstance(trans_metadata, pd.DataFrame)