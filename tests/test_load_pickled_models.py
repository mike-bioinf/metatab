import pickle
from pathlib import Path
from typing import Any
from sklearn.datasets import load_iris


def try_test_model_on_iris(filename: str):
    X, y = load_iris(return_X_y=True, as_frame=True)
    try:
        model = load_model(filename)
        _ = model.predict_proba(X)
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


def test_my_es_randomized_xgb_classifier():
    try_test_model_on_iris("my_es_randomized_xgb_classifier.pkl")


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