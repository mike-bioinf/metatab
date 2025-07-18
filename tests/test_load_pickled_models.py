import pickle
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.datasets import load_iris
from estimators.types import Estimator


def try_test_model_on_iris(filename: str):
    X, y = load_iris(return_X_y=True, as_frame=True)
    try:
        model: Estimator = load_model(filename)
        pred_proba = model.predict_proba(X)
        if not isinstance(pred_proba, np.ndarray):
            raise TypeError("The estimator does not predict numpy arrays.")
    except Exception:
        assert False, f"Problem when loading and/or using the '{filename}' model."


def load_model(filemodel: str | Path) -> Any:
    '''Utility to deserialize the model from the pkl filename'''
    folder = Path(__file__).parents[1] / "fitted_models/fitted_models"
    model_path = folder / filemodel
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def test_my_randomized_xgb_classifier():
    try_test_model_on_iris("my_randomized_xgb_classifier.pkl")


def test_my_randomized_es_xgb_classifier():
    try_test_model_on_iris("my_randomized_es_xgb_classifier.pkl")


def test_my_xgb_classifier():
    try_test_model_on_iris("my_xgb_classifier.pkl")


def test_my_es_xgb_classifier():
    try_test_model_on_iris("my_es_xgb_classifier.pkl")


def test_my_rf_classifier():
    try_test_model_on_iris("my_rf_classifier.pkl")


def test_my_randomized_rf_classifier():
    try_test_model_on_iris("my_randomized_rf_classifier.pkl")


def test_my_tabfn_classifier():
    try_test_model_on_iris("my_tabpfn_classifier.pkl")