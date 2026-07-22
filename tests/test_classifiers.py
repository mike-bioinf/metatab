"""
This module contains a battery of test executed on the classifiers used in this package.
The classifiers are from third party packages or wrappers around them.
This modules aims to verify a set of expectations about the classiiers on which we rely.
"""

import pytest
import warnings
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from metatab.estimators.estimators.realmlp import RealMLPClassifier
from metatab.estimators.estimators.tabm import TabMClassifier
from metatab.estimators.estimators.catboost import CatBoostClassifierInterface
from autogluon.tabular import TabularPredictor


classifier_classes = [
    RandomForestClassifier,
    ExtraTreesClassifier, 
    XGBClassifier, 
    LGBMClassifier, 
    CatBoostClassifierInterface,
    TabPFNClassifier,
    RealMLPClassifier,
    TabMClassifier,
]


@pytest.fixture(scope="module")
def get_iris_sets() -> tuple:
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, train_size=0.3, random_state=0, stratify=y)


@pytest.mark.parametrize("classifier_class", classifier_classes)
def test_set_params_method(classifier_class):
    '''Test that all classifiers have a working set_params method'''
    clf = classifier_class()
    # we set the "random_state" parameter since is shared by all classifiers
    clf.set_params(random_state=1234)
    assert clf.get_params()["random_state"] == 1234, "'set_params' method is not properly implemented for the classifier"


@pytest.mark.parametrize("classifier_class", classifier_classes)
def test_that_classifiers_learns_sklearn_attributes_as_expected(classifier_class, get_iris_sets):
    '''
    Test that all classifiers learns the expected sklearn attributes.
    In particular we check that the target integer-encoded labels are "learned" in a sorted increasing order.
    This is essential to assure "compability" between the different classifier predictions.
    '''
    X_train, X_val, y_train, y_val = get_iris_sets
    expected_classes = np.sort(y_train.unique())
    
    init_args = {}
    if classifier_class is CatBoostClassifierInterface:
        init_args = dict(allow_writing_files=False)
    
    clf = classifier_class(**init_args)

    extra_fit_args = {}
    if isinstance(clf, (TabMClassifier, RealMLPClassifier)):
        extra_fit_args = dict(eval_set=[(X_val, y_val)])
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6.*",
            category=FutureWarning
        )
        clf.fit(X_train, y_train, **extra_fit_args)
    
    assert getattr(clf, "classes_", None) is not None, "The classifier does not learn the 'classes_' attribute."
    assert (clf.classes_ == expected_classes).all(), "The classifier does not learn integer-encoded labels in a increasing sorted order."
    ## NOT NECESSARY
    #assert hasattr(clf, "n_features_in_"), "The classifier does not learn the 'n_features_in_' attribute."
    #assert hasattr(clf, "feature_names_in_"), "The classifier does not learn the 'feature_names_in_' attribute."



def test_that_autogluon_learns_expected_class_order():
    X, y = load_iris(return_X_y=True, as_frame=True)
    data = pd.concat([X, y], axis=1)
    expected_classes = np.sort(y.unique())

    out_path = Path(__file__).parent / "autogluon_test_folder"
    predictor = TabularPredictor(label="target", path=str(out_path), verbosity=0)

    try:
        predictor.fit(data, presets="medium_quality", time_limit=30)
    except Exception as e:
        shutil.rmtree(out_path)
        raise

    shutil.rmtree(out_path)

    assert (predictor.classes_ == expected_classes).all(), "Autogluon does not learn integer-encoded labels in a increasing sorted order." 