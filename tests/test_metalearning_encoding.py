import pytest
import pandas as pd
from pathlib import Path
from typing import Literal
from sklearn.pipeline import make_pipeline
from metalearning.encoding import get_encoding_scheme



def load_metadata(
    estimator: Literal["random_forest", "xgb", "catboost", "lgbm", "tabpfn"],
    preprocessing: Literal["base", "density_filter", "pca"] = "base"
) -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "data/metadata" / f"{estimator}.txt", sep="\t")
    df["preprocessing"] = preprocessing
    return df



## TODO: catboost is missing since the metadata is missing
## TODO: tabpfn is not working, we have to decide the metadata "state" (corrected ot not)
LIST_META_ESTIMATORS = [
    "random_forest",
    "xgb",
    "lgbm",
    "catboost",
    "tabpfn"
]


@pytest.mark.parametrize("estimator", LIST_META_ESTIMATORS)
def test_that_estimator_metadata_encoding_scheme_is_correct(estimator):
    enc_pipe = make_pipeline(*get_encoding_scheme(estimator))
    enc_pipe.fit_transform(load_metadata(estimator))
