import numpy as np
import pandas as pd
from functools import partial
from sklearn.datasets import make_classification
from metatab.estimators.params import TuningParams
from metatab.hp_search.point_corrector import PointCorrector
from metatab.metalearning.metadata_evaluator import MetadataEvaluator
from metatab.metalearning.sampler import HyperoptRandomSampler
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.metadata_generator import MetadataGenerator
from metatab.metalearning.acquisition_funcs import compute_upper_confidence_bound
from metatab.metalearning.load import query_surrogate_framework



DUMMY_METADATA = pd.DataFrame(np.array([[2, 1, 0, 4, 10, 7, 99, 18]]).T, columns=["col_0"])
DUMMY_CANDIDATE_POINTS = ["sixth", "seventh", "eighth", "fifth", "third", "fourth", "first", "second"]


class DummySurrogateFramework:
    def predict(self, metadata: pd.DataFrame):
        return metadata["col_0"].to_numpy(), np.zeros_like(metadata.shape[0])


def dummy_acquisition_func(value: np.ndarray, uncertanty: np.ndarray, *args, **kwargs):
    return value + uncertanty


def create_dummy_metadata_evaluator():
    metadata_evaluator = MetadataEvaluator(
        surrogate_framework=DummySurrogateFramework(),
        acquisition_func=dummy_acquisition_func
    )
    metadata_evaluator.fit(DUMMY_METADATA, DUMMY_CANDIDATE_POINTS)
    metadata_evaluator.evaluate_candidates()
    return metadata_evaluator



def test_metadata_evaluator_propose_n_best_method():
    metadata_evaluator = create_dummy_metadata_evaluator()
    points = metadata_evaluator.propose_n_best(n_best=3)
    assert points == ["first", "second", "third"], "propose_n_best returns the wrong points"



def test_metadata_evaluator_propose_uniform_from_top():
    metadata_evaluator = create_dummy_metadata_evaluator()
    points = metadata_evaluator.propose_uniform_from_top(n_steps=2, step_size=3)
    assert points == ["first", "fourth"], "propose_best_uniform returns the wrong points"



def test_metadata_evaluator_propose_random_from_top_method():
    metadata_evaluator = create_dummy_metadata_evaluator()

    points = metadata_evaluator.propose_random_from_top(
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


def test_metadata_evaluator_works_in_real_scenario():
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
        mfe=CustomMFE()
    )

    meta_generator.fit(X, y, TuningParams.LGMB_C0, seed=0)

    metadata, candidate_points = meta_generator.generate(
        n_points=10,
        mfe_extract_kwargs={"add_features": {"preprocessing": "base"}} 
    )

    metadata_evaluator = MetadataEvaluator(
        surrogate_framework=surrogate_framework,
        acquisition_func=partial_compute_upper_confidence_bound
    )

    _ = metadata_evaluator.fit(metadata, candidate_points).evaluate_candidates()
    best_points = metadata_evaluator.propose_n_best(n_best=2)

    assert isinstance(best_points, list), "The surrogate worker does not return a list."
    assert len(best_points) == 2, "The surrogate worker select a wrong number of points."
    
    for el in best_points:
        assert isinstance(el, dict), "The surrogate worker propose wrong typed points."
        assert "learning_rate" in el.keys(), "The surrogate worker sampler is not returning the right hps."