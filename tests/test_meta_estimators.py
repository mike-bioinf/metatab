import pytest
import pandas as pd
from sklearn.datasets import load_iris
from metatab.metalearning.utils import BestMetaStrategyParams

from metatab.estimators import (
    MetaTuneRandomForestClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier,
    MetaTuneTabPFNClassifier,
    MetaEnsembleRandomForestClassifier,
    MetaEnsembleXGBClassifier,
    MetaEnsembleEsXGBClassifier,
    MetaEnsembleTabPFNClassifier,
    MetaEnsembleLGBMClassifier,
    MetaEnsembleEsLGBMClassifier
)


METATUNECLASSES = [
    MetaTuneRandomForestClassifier,
    MetaTuneXGBClassifier,
    MetaTuneEsXGBClassifier,
    MetaTuneLGBMClassifier,
    MetaTuneEsLGBMClassifier,
    MetaTuneTabPFNClassifier
]


METAENSCLASSES = [
    MetaEnsembleRandomForestClassifier,
    MetaEnsembleXGBClassifier,
    MetaEnsembleEsXGBClassifier,
    MetaEnsembleLGBMClassifier,
    MetaEnsembleEsLGBMClassifier,
    MetaEnsembleTabPFNClassifier
]


@pytest.mark.parametrize("metaclass", METATUNECLASSES)
def test_metatune_estimator(metaclass):
    X, y = load_iris(return_X_y=True, as_frame=True)
    
    meta_estimator = metaclass(
        n_iter=2,
        n_cv_folds=2,
        meta_strategy_params=BestMetaStrategyParams(n_candidate_points=3),
        build_df_search=True
    )
    
    _ = meta_estimator.fit(X, y).predict_proba(X)

    assert meta_estimator.search_losses_.size == 2, "The metaclass evaluates the wrong number of points."
    assert hasattr(meta_estimator, "df_search_")
    assert isinstance(meta_estimator.df_search_, pd.DataFrame)


## Test for no errors
@pytest.mark.parametrize("metaclass", METAENSCLASSES)
def test_metaens_estimator(metaclass, tmp_path_factory):
    X, y = load_iris(return_X_y=True, as_frame=True)

    meta_estimator = metaclass(
        save_path=tmp_path_factory.mktemp("generic_folder"),
        n_members=1,
        meta_strategy="best",
        meta_strategy_params=BestMetaStrategyParams(n_candidate_points=100),
        log=50
    )

    _ = meta_estimator.fit(X, y).predict_proba(X)