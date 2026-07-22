import pytest
from sklearn.datasets import make_classification
from metatab.metatab_utils.exceptions import DeviceError
from metatab.ensemble.configuration import UserEnsembleConfiguration
from metatab.estimators.estimators.rf import MyRandomForestClassifier



def test_that_cuda_does_not_work_with_incompatible_estimators():
    X, y = make_classification()
    rf = MyRandomForestClassifier(preprocessing="base", seed=0, n_threads=1, device="cuda")

    with pytest.raises(expected_exception=DeviceError):
        rf.fit(X, y)

    with pytest.raises(expected_exception=DeviceError):
        UserEnsembleConfiguration(
            name="conf",
            algo="random",
            n_members=1,
            estimator="random_forest",
            preprocessing="estimator_default",
            tune_space="default",
            early_stop_on_validation_set=False,
            device="cuda"
        )
    

## test for no error
def test_that_device_auto_option_works_correctly():
    X, y = make_classification(n_samples=10, n_features=4)
    rf = MyRandomForestClassifier(preprocessing="base", seed=0, n_threads=1, device="auto")
    rf.fit(X, y)

    UserEnsembleConfiguration(
        name="conf",
        algo="random",
        n_members=1,
        estimator="random_forest",
        preprocessing="estimator_default",
        tune_space="default",
        early_stop_on_validation_set=False,
        device="auto"
    )