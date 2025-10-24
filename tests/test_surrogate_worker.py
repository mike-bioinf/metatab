import pytest
import pandas as pd
from functools import partial
from sklearn.datasets import make_classification
from estimators.params import TuningParams
from hp_search.point_corrector import PointCorrector
from metalearning.surrogate_worker import SurrogateWorker
from metalearning.sampler import HyperoptRandomSampler
from metalearning.metafeatures import CustomMFE
from metalearning.acquisition_funcs import compute_upper_confidence_bound
from metalearning.database.utils import query_surrogate_framework



@pytest.fixture(scope="module")
def create_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(random_state=0)
    X = pd.DataFrame(X)
    X.columns = pd.Series([f"col_{i}" for i in range(X.columns.size)])
    y = pd.Series(y)
    return X, y



def test_surrogate_worker_works_in_general(create_data):
    X, y = create_data
    surrogate_framework = query_surrogate_framework("lgbm")

    partial_compute_upper_confidence_bound = partial(
        compute_upper_confidence_bound,
        k=1,
        mean_direction="lower_is_better"
    )

    surrogate_worker = SurrogateWorker(
        sampler=HyperoptRandomSampler(),
        mfe=CustomMFE(seed=0),
        point_corrector=PointCorrector(),
        surrogate_framework=surrogate_framework,
        acquisition_func=partial_compute_upper_confidence_bound
    )

    best_points = surrogate_worker.fit(X, y, TuningParams.LGMB_C0, seed=0).propose_n_best(
        n_candidate_points=10,
        n_best=2,
        mfe_extract_kwargs={"add_features": {"preprocessing": "base"}}
    )

    assert isinstance(best_points, list), "The surrogate worker does not return a list."
    assert len(best_points) == 2, "The surrogate worker select a wrong number of points."
    
    for el in best_points:
        assert isinstance(el, dict), "The surrogate worker propose wrong typed points."
        assert "learning_rate" in el.keys(), "The surrogate worker sampler is not returning the right hps."
    