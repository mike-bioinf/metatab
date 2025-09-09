import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from estimators import Estimator



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



## Random forest
def test_my_rf_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_rf_classifier.pkl")

def test_my_tuned_rf_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_rf_classifier.pkl")


## XGB
def test_my_xgb_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_xgb_classifier.pkl")

def test_my_es_xgb_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_es_xgb_classifier.pkl")

def test_my_tuned_xgb_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_xgb_classifier.pkl")

def test_my_tuned_es_xgb_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_es_xgb_classifier.pkl")


## Catboost
def test_my_catboost_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_catboost_classifier.pkl")

def test_my_es_catboost_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_es_catboost_classifier.pkl")

def test_my_tuned_catboost_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_catboost_classifier.pkl")

def test_my_tuned_es_catboost_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_es_catboost_classifier.pkl")


## LGBM
def test_my_lgbm_classifier(fit_estimators_on_iris):
     try_test_model_on_iris(fit_estimators_on_iris / "my_lgbm_classifier.pkl")

def test_my_es_lgbm_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_es_lgbm_classifier.pkl")

def test_my_tuned_lgbm_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_lgbm_classifier.pkl")

def test_my_tuned_es_lgbm_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_es_lgbm_classifier.pkl")


## tabpfn
def test_my_tabpfn_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tabpfn_classifier.pkl")

def test_my_tuned_tabpfn_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_tuned_tabpfn_classifier.pkl")

def test_my_aesfinetuned_tabpfn_classifier(fit_estimators_on_iris):
    try_test_model_on_iris(fit_estimators_on_iris / "my_aesfinetunedtabpfn_classifier.pkl")