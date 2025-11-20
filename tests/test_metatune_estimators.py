import pytest
import pandas as pd
from sklearn.datasets import load_iris
from hp_search.utils import BestMetaStrategyParams

from estimators import (
    MetaTuneRandomForestClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier,
    MetaTuneTabPFNClassifier
)



METACLASSES = [
    MetaTuneRandomForestClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier,
    MetaTuneTabPFNClassifier
]


@pytest.mark.parametrize("metaclass", METACLASSES)
def test_metatune_estimator(metaclass):
    X, y = load_iris(return_X_y=True, as_frame=True)
    
    meta_estimator = metaclass(
        n_iter=2,
        n_cv_folds=2,
        meta_strategy_params=BestMetaStrategyParams(n_candidate_points=3),
        build_df_search=True
    )
    
    _ = meta_estimator.fit(X, y).predict_proba(X)

    assert meta_estimator.search_losses_.size == 2, "The metaclass evaluates the wrong number of points"
    assert hasattr(meta_estimator, "df_search_")
    assert isinstance(meta_estimator.df_search_, pd.DataFrame)