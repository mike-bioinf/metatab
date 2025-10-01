import pytest
import numpy as np
import pandas as pd
from typing import Literal
from metatab_utils.prediction import PredictionDataframe



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



def create_rows_to_add(n_rows: int):
    rng = np.random.default_rng(100)
    rows = []
    for i in range(n_rows):
        rows.append({
            "dataset": f"dataset_{i}",
            "pred_proba": softmax(rng.normal(size=(2, 2))),
            "y_train": rng.integers(low=0, high=1, endpoint=True, size=10),
            "y_test": np.array([0, 1])
        })    
    return rows



def test_add_rows_works_on_existing_df():
    pred_df = PredictionDataframe()
    pred_df.build_from_data(*create_data())

    single_row_to_add = create_rows_to_add(1)
    two_rows_to_add = create_rows_to_add(2)

    pred_df.add_rows(single_row_to_add, compute_metrics=True, multiclass="average", average_strategy="macro")

    assert pred_df.df.shape[0] == 3, "Number of rows after single row addition is wrong."
    assert pred_df.df["auc"].isna().sum() == 2, "Error in performance metrics computation or concatenation."

    pred_df.add_rows(two_rows_to_add)
    assert pred_df.df.shape[0] == 5, "Number of rows after multiple rows addition is wrong."
    assert pred_df.df["auc"].isna().sum() == 4, "Error in performance metrics computation or concatenation."  



def test_add_rows_build_the_dataframe_if_missing():
    pred_df = PredictionDataframe()
    single_row_to_add = create_rows_to_add(1)
    pred_df.add_rows(single_row_to_add, compute_metrics=False)
    assert isinstance(pred_df.df, pd.DataFrame), "add_rows is not able to build the dataframe when it is missing."
    assert pred_df.df.shape[0] == 1, "Problems inthe underlying dataframe when adding rows from nothing."