import pytest
import pickle
import numpy as np
from pathlib import Path
from typing import Literal
from sklearn.datasets import load_iris
from metatab.estimators.estimators import Estimator
from metatab.hp_search.searchcv import SearchCV

from tests.conftest import (
    ESTIMATOR_TUNE_CONFIGS, 
    ESTIMATOR_ALL_CONFIGS,
    get_alternative_estimator_file_names
)



def try_test_model_on_iris(estimator_path: Path):
    X, y = load_iris(return_X_y=True, as_frame=True)
    try:
        with open(estimator_path, "rb") as f:
            estimator: Estimator = pickle.load(f)
        pred_proba = estimator.predict_proba(X)
        if not isinstance(pred_proba, np.ndarray):
            raise TypeError("The estimator does not predict numpy arrays.")
    except Exception:
        estimator_filename = estimator_path.stem
        assert False, f"Problem when loading and/or using the '{estimator_filename}' model."



def check_nan_in_hpo_losses(estimator_path: Path) -> None:
    if verify_presence_nan_in_hpo_losses(estimator_path, returns="bool"):
        raise ValueError(f"Found np.nan values in the search losses of the model: {estimator_path}")



def verify_presence_nan_in_hpo_losses(
    estimator_path: Path, 
    returns: Literal["bool", "int", "both"]
) -> bool|int|tuple[bool,int]:
    '''
    Checks on the presence of nan values in the search_losses_ of SearchCV instance.
    When returns is "bool", it returns a bool indicating the presence of nan values.
    When returns is "int", it returns the number of nan values.
    When returns is "both", it returns both as a tuple [bool, int].
    '''
    with open(estimator_path, "rb") as f:
        estimator: Estimator = pickle.load(f)

    assert hasattr(estimator, "estimator_")
    assert isinstance(estimator.estimator_, SearchCV)
    assert hasattr(estimator.estimator_, "search_losses_")
    
    losses = np.array(estimator.estimator_.search_losses_)
    nan_losses = np.isnan(losses)
    number_of_nans = nan_losses.sum()
    is_nan_present = nan_losses.any()

    if returns == "bool":
        result = is_nan_present
    elif returns == "int":
        result = number_of_nans
    elif returns == "both":
        result [is_nan_present, number_of_nans]
    else:
        raise ValueError(f"returns cannot be equal to '{returns}'.")
    
    return result



@pytest.mark.parametrize("fitted_model", list(ESTIMATOR_ALL_CONFIGS.keys()))
def test_fitted_estimators_on_iris(fitted_model, fit_estimators_on_iris):
    '''
    Test that the estimators fitted on the iris datasets work,
    meaning that they are able to give predictions.
    '''
    try_test_model_on_iris(fit_estimators_on_iris / fitted_model)



@pytest.mark.parametrize("fitted_tuned_model", ESTIMATOR_TUNE_CONFIGS.keys())
def test_nan_values_in_search_losses(fitted_tuned_model, fit_estimators_on_iris):
    '''
    Test that in the tuning process no point evaluation fails.
    This is a good general indication whether the tune space is set up correctly.
    '''
    check_nan_in_hpo_losses(fit_estimators_on_iris / fitted_tuned_model)



@pytest.mark.parametrize("fitted_alternative_tuned_model", get_alternative_estimator_file_names())
def test_nan_values_in_search_losses_for_alternative_tune_spaces(fitted_alternative_tuned_model, fit_estimators_alternative_tune_configs):
    '''
    Test that in the tuning process of the alternative spaces no point evaluation fails.
    This is a good general indication whether the tune space is set up correctly.
    '''
    check_nan_in_hpo_losses(fit_estimators_alternative_tune_configs / fitted_alternative_tuned_model)