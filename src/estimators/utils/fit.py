import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from estimators.utils.general import remove_string_from_params
from estimators.utils.types import Classifier



def fit_with_early_stop_on_validation_set(
    *,
    clf_or_pipe: Classifier | Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    validation_set_size: float,
    eval_set_parameter: str,
    fit_classifier_kwargs: dict,
    return_fit_time: bool = False
 ) -> Classifier | Pipeline | tuple[Classifier|Pipeline, float]:
    '''
    Utility to fit an estimator using early stop on a validation set.
    The estimator must implement the early stop capability at its 
    fit interface, following a GBDT-like API ("eval_set-like" parameter).

    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            The classifier or pipeline to fit. 
            If a pipeline it must ends with a classifier.
        
        X (pd.DataFrame): Training feature space.
        
        y (pd.Series): Training labels.
        
        seed (int): Seed for reproducibility used ONLY in the train/val splitting.
        
        validation_set_size (float): 
            Ratio of training data to use as validation.
            Must be a number in (0, 1).
        
        eval_set_parameter (str): 
            Name of the parameter accepting the validation sets.

        fit_classifier_kwargs (dict):
            A dict unpackaged in the classifier fit calls.
            The dict keys can be either in classifier or pipeline "formats" 
            indipendently of the `clf_or_pipe` object.

        return_fit_time (bool, optional):
            Whether to return the fit time along the fitted clf_or_pipe.
            If True returns a tuple [clf_or_pipe, fit_time], otherwise clf_or_pipe directly.

    Returns:
        Classifier|Pipeline|tuple: 
        The fitted estimator alone or in a tuple with the fit time.
    '''
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=validation_set_size,
        random_state=seed,
        stratify=y
    )
    
    # since we pop the classifier from the pipeline we must remove 
    # the classifier name from the fit_kwargs keys in every scenario
    fit_classifier_kwargs = remove_string_from_params(
        params_dict=fit_classifier_kwargs, 
        string=f"{clf_or_pipe.steps[-1][0]}__"
    )

    # we always consider the preprocessing in the fit time
    start_fit_time = time.time()

    if isinstance(clf_or_pipe, Pipeline):
        # we split the classifier from the preprocessing pipeline 
        # to avoid to repeat the preprocessing 2 times.
        # we fit in place the two components separately.
        clf: Classifier = clf_or_pipe[-1]
        preprocessing_pipeline: Pipeline = clf_or_pipe[:-1]
        X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
        X_val_transformed = preprocessing_pipeline.transform(X_val)

        clf.fit(
            X_train_transformed, y_train, 
            **{eval_set_parameter: [(X_val_transformed, y_val)]},
            **fit_classifier_kwargs
        )

    else:
        clf_or_pipe.fit(
            X_train, y_train,
            **{eval_set_parameter: [(X_val, y_val)]},
            **fit_classifier_kwargs
        )
    
    fit_time = time.time() - start_fit_time

    if return_fit_time:
        return [clf_or_pipe, fit_time]
    else:
        return clf_or_pipe
