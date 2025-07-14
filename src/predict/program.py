'''
Collections of funcs that will form the main program
'''

from __future__ import annotations

from sklearn.model_selection import RandomizedSearchCV
from typing import TYPE_CHECKING
from sklearn.pipeline import Pipeline

from estimators import(
    MyESRandomizedXGBClassifier,
    MyRandomizedXGBClassifier,
    MyRandomSearchCV
)

if TYPE_CHECKING:
    import numpy as np



def check_type_deserialized_object(obj) -> None:
    '''Check whether the pickle deserialized object type is expected'''
    if not isinstance(
        obj, 
        (
            MyESRandomizedXGBClassifier,
            MyRandomizedXGBClassifier
        )
    ):
        raise TypeError("The deserialized object is not an estimator type-wise.")
    


def check_that_object_is_fitted(obj) -> None:
    '''Check whether the object has the "estimator_" attribute.'''
    if not hasattr(obj, "estimator_"):
        raise ValueError(
            "The object does not contain a fitted estimator ('estimator_' attribute)."
        )



def take_feature_names_in_attr(obj) -> np.ndarray:
    '''
    Retrieve the "feature_names_in_" attribute from the 
    fitted estimator accordingly to his type.

    Note that we expect this attribute to be present 
    since we fit the estimators on pandas DataFrames.
    
    Note that the fitted estimator is the one
    learned inside our wrapper classes ("estimator_" attr).
    
    In detail we expect 4 possible types: RandomizedSearchCV,
    MyRandomSearchCV, Pipeline and sklearn or sklearn-like estimators.
    '''
    fitted_estimator = obj.estimator_
    
    if isinstance(fitted_estimator, RandomizedSearchCV):
        return fitted_estimator.best_estimator_.feature_names_in_
    
    elif isinstance(fitted_estimator, MyRandomSearchCV):
        ## TODO: This is valid for xgboost but must be checked for CATBOOST (and eventually others)
        return fitted_estimator.best_clf_.feature_names_in_
    
    else:
        # pipelines and sklearn-like estimators expose directly the attr
        return fitted_estimator.feature_names_in_



def reindex_X_test():
    ## TODO: check that the reindex does not bring a full 0 df
    ## meaning probably no feature in common with fitted df
    pass


