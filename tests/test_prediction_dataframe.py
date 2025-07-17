import pytest
import numpy as np
from typing import Literal
from utils.prediction import PredictionDataframe



def softmax(x: np.ndarray):
    '''Softmax on 2d array'''
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def create_data(error_type: Literal["length", "dimension", "shape", "na"] | None = None) -> tuple:
    '''
    Create the data needed to build the prediction dataframe 
    optionally with different type of errors.
    '''
    rng = np.random.default_rng(100)
    dataset = "first_dataset" if error_type  == "length" else ["first_dataset", "second_dataset"]

    if error_type == "dimension":
        y_train = [
            rng.integers(low=0, high=2, size=(30, 2)), 
            rng.integers(low=0, high=3, size=45)
        ]
    else:
        y_train = [
            rng.integers(low=0, high=2, size=30), 
            rng.integers(low=0, high=3, size=45)
        ]
    
    if error_type == "shape":
        y_test = [
            rng.integers(low=0, high=2, size=1000), 
            rng.integers(low=0, high=3, size=20)
        ]
    else:
        y_test = [
            rng.integers(low=0, high=2, size=10), 
            rng.integers(low=0, high=3, size=20)
        ]
    
    pred_proba = [
        softmax(rng.normal(size=(10, 2))), 
        softmax(rng.normal(size=(20, 3)))
    ]
    
    if error_type == "na":
        pred_proba[0] = np.nan
    
    return dataset, y_train, y_test, pred_proba



def test_build_method_works():
    dataset, y_train, y_test, pred_proba = create_data()
    pred_df = PredictionDataframe()
    pred_df.build_from_data(dataset, y_train, y_test, pred_proba, sup_col="additional")



def test_build_method_raise_expections():
    pred_df = PredictionDataframe()

    with pytest.raises(Exception, match="The input iterables have not the same length"):
        pred_df.build_from_data(*create_data("length"))

    with pytest.raises(Exception):
        pred_df.build_from_data(*create_data(), test_labels=22)

    with pytest.raises(Exception, match="Not all arrays in"):
        pred_df.build_from_data(*create_data("dimension"))

    with pytest.raises(Exception, match="Found discrepancies in the shapes of 'y_test' and 'pred_proba' arrays"):
        pred_df.build_from_data(*create_data("shape"))



def test_build_method_works_with_na():
    pred_df = PredictionDataframe()
    pred_df.build_from_data(*create_data("na"))
