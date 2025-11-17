import numpy as np
import pandas as pd
from functools import partial
from sklearn.datasets import make_classification
from estimators.params import TuningParams
from hp_search.point_corrector import PointCorrector
from metalearning.surrogate_worker import SurrogateWorker
from metalearning.sampler import HyperoptRandomSampler
from metalearning.metafeatures import CustomMFE
from metalearning.generator import MetadataGenerator
from metalearning.acquisition_funcs import compute_upper_confidence_bound
from metalearning.load import query_surrogate_framework



class DummyMetadataGenerator:
    def fit(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        # we craft a list of values and relative positions in the decreasing order
        metadata = pd.DataFrame(np.array([[2, 1, 0, 4, 10, 7, 99, 18]]).T, columns=["col_0"])
        canditate_points = ["sixth", "seventh", "eighth", "fifth", "third", "fourth", "first", "second"]
        return metadata, canditate_points


class DummySurrogateFramework:
    def predict(self, metadata: pd.DataFrame):
        return metadata["col_0"].to_numpy(), np.zeros_like(metadata.shape[0])


def dummy_acquisition_func(value: np.ndarray, uncertanty: np.ndarray, *args, **kwargs):
    return value + uncertanty


def create_dummy_surrogate_worker():
    surrogate_worker = SurrogateWorker(
        metadata_generator=DummyMetadataGenerator(),
        surrogate_framework=DummySurrogateFramework(),
        acquisition_func=dummy_acquisition_func
    )
    surrogate_worker.is_fitted_ = True
    return surrogate_worker



def test_surrogate_worker_propose_n_best_method():
    surrogate_worker = create_dummy_surrogate_worker()

    points = surrogate_worker.fit(0, 1, 2, 3).propose_n_best(
        n_candidate_points=100, # ignored but must be greater than n_best
        n_best=3
    )

    assert points == ["first", "second", "third"], "propose_n_best returns the wrong points"


def test_surrogate_worker_propose_uniform_from_top():
    surrogate_worker = create_dummy_surrogate_worker()

    points = surrogate_worker.propose_uniform_from_top(
        n_candidate_points=1000, # ignored but must be greater than "n_steps * step_size"
        n_steps=2,
        step_size=3
    )
    
    assert points == ["first", "fourth"], "propose_best_uniform returns the wrong points"


def test_surrogate_worker_propose_random_from_top_method():
    surrogate_worker = create_dummy_surrogate_worker()

    points = surrogate_worker.propose_random_from_top(
        n_candidate_points=1000, # ignored but must be greater than n_proposed and top
        n_proposed=4,
        top=8, # include all points
        seed=0
    )
    
    rng = np.random.default_rng(0)
    
    expected_points = rng.choice(
        # we use the ordered list since the method internally orders from best to worst
        ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"],
        size=4,
        replace=False
    )

    assert (np.array(points) == expected_points).all(), "propose_random_from_top returns the wrong points"



def create_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(random_state=0)
    X = pd.DataFrame(X)
    X.columns = pd.Series([f"col_{i}" for i in range(X.columns.size)])
    y = pd.Series(y)
    return X, y


def test_surrogate_worker_works_in_a_real_scenario():
    X, y = create_data()
    surrogate_framework = query_surrogate_framework("lgbm")

    partial_compute_upper_confidence_bound = partial(
        compute_upper_confidence_bound,
        k=1,
        mean_direction="lower_is_better"
    )

    meta_generator = MetadataGenerator(
        sampler=HyperoptRandomSampler(),
        point_corrector=PointCorrector(),
        mfe=CustomMFE(seed=0)
    )

    surrogate_worker = SurrogateWorker(
        metadata_generator=meta_generator,
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