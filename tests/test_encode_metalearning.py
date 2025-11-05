import pytest
import pandas as pd
from pathlib import Path
from typing import Literal
from sklearn.pipeline import make_pipeline
from estimators.types import TUNABLE_ESTIMATOR_TYPE
from metalearning.encode.encode import get_encoding_scheme
from metalearning.encode.transformers import NanToNone, ColToStr



def load_metadata(
    estimator: TUNABLE_ESTIMATOR_TYPE,
    preprocessing: Literal["base", "density_filter", "pca"] = "base"
) -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "data/metadata" / f"{estimator}.txt", sep="\t")
    df["preprocessing"] = preprocessing
    return df



def test_that_nan_to_none_transformer_works():
    metadata = load_metadata("tabpfn")
    assert metadata["inference_config__OUTLIER_REMOVAL_STD"].isna().any()
    nan_to_none = NanToNone("inference_config__OUTLIER_REMOVAL_STD", check_on_fit=False).set_output(transform="pandas")
    # test that set_output framework works
    trans_metadata = nan_to_none.fit_transform(metadata)
    assert isinstance(trans_metadata, pd.DataFrame)
    assert (trans_metadata.columns == metadata.columns).all(), "NoneToNone transformer changes column names"
    # test that nantonone converts nan to None
    is_none_list = [el is None for el in trans_metadata["inference_config__OUTLIER_REMOVAL_STD"]]
    is_nan_list = [pd.isna(el) for el in trans_metadata["inference_config__OUTLIER_REMOVAL_STD"] if el is not None]
    assert any(is_none_list), "NanToNone transformer is not converting nan to None"
    assert not any(is_nan_list), "NanToNone transformer is not converting nan to None"



def test_that_col_to_str_transformer_works():
    metadata = load_metadata("tabpfn")
    assert pd.api.types.is_numeric_dtype(metadata["loss"].dtype)
    col_to_str = ColToStr("loss", check_on_fit=False).set_output(transform="pandas")
    trans_metadata = col_to_str.fit_transform(metadata)
    # test that set_output framework works
    assert isinstance(trans_metadata, pd.DataFrame)
    assert (metadata.columns == trans_metadata.columns).all(), "ColToStr transformer chnages column names"
    # test that cthe transformer convert to str
    assert pd.api.types.is_object_dtype(trans_metadata["loss"].dtype)    
    for value in trans_metadata["loss"]:
        assert isinstance(value, str), "ColToStr doesn't cast to str."



@pytest.mark.parametrize("estimator", ["random_forest", "xgb", "lgbm", "catboost", "tabpfn"])
def test_that_estimator_metadata_encoding_scheme_is_correct(estimator):
    enc_pipe = make_pipeline(*get_encoding_scheme(estimator)).set_output(transform="pandas")
    trans_metadata = enc_pipe.fit_transform(load_metadata(estimator))
    assert isinstance(trans_metadata, pd.DataFrame)