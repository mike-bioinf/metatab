import time
import pandas as pd
from copy import deepcopy
from numpy.random import RandomState
from typing import Literal, Any
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from estimators.constants import Classifier
from preprocessing import DensityFeatureSelector



def get_fresh_random_state(random_state: None | int | RandomState) -> RandomState:
    '''
    Get a fresh random state instance.
    If the input is None it generates a new instance seeded with 0.
    If int then it produces the random state using it as seed.
    If RandomState it returns a deepcopy of it.
    '''
    if random_state is None:
        return RandomState(0)
    elif isinstance(random_state, int):
        return RandomState(random_state)
    elif isinstance(random_state, RandomState):
        return deepcopy(random_state)
    else:
        raise ValueError("Unsupported input.")



def add_string_to_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to add at the beginning of dict keys a string.
    This is helpful when using sklearn pipelines.
    Notes that the function assumes that the keys are str.
    Returns a new dict.
    '''
    return {f"{string}{k}":v for k, v in params_dict.items()}



def remove_string_from_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to remove the string from the beginning of params dict keys.
    Notes that the function assumes that the keys are of str type.
    Returns a new dict.
    '''
    new_params_dict = {}
    for key, value in params_dict.items():
        new_key = key.removeprefix(string)
        new_params_dict[new_key] = value
    return new_params_dict



def update_dict( 
    dictionary: dict, 
    name_key: str, 
    value: Any, 
    copy: bool = False
) -> dict:
    '''
    Update the dict or a deepcopy of it with the name_key:value couple.
    Returns the updated dict.
    '''
    dictionary = deepcopy(dictionary) if copy else dictionary
    dictionary[name_key] = value
    return dictionary



def create_default_pipeline(
    preprocessing:  Literal["base", "density_filter", "pca"],
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None,
) -> Pipeline:
    '''
    Creates the standard pipelines for each preprocessing strategy.
    Allows to adapt the dinamic strategy parameter of the DenistyFeatureSelector.
    Allows to use the classifier as final step or not. 
    If used one must specify also the parameters to pass in it.
    '''
    if preprocessing == "base":
        return create_base_default_pipeline(classifier, classifier_params)
    elif preprocessing == "pca":
        return create_pca_default_pipeline(classifier, classifier_params)
    elif preprocessing == "density_filter":
        return create_density_filter_default_pipeline(
            density_feature_selector_strategy,
            classifier,
            classifier_params
        )
    else:
        raise ValueError("Unsupported preprocessing.")



def create_base_default_pipeline(
    classifier: Classifier | None = None,
    classifier_params: dict | None = None
) -> Pipeline:
    '''
    Creates the default pipeline in case of "base" preprocessing.
    Allows to control whether to add ot not the classifier as last step.
    '''
    return make_pipeline(
        *add_classifier_head_to_steps(
            (VarianceThreshold(),), 
            classifier, 
            classifier_params
        )
    )


def create_pca_default_pipeline(
    classifier: Classifier | None = None,
    classifier_params: dict | None = None  
) -> Pipeline:
    '''
    Creates the default pipeline in case of "pca" preprocessing.
    Allows to control whether to add ot not the classifier as last step.
    '''
    return make_pipeline(
        *add_classifier_head_to_steps(
            (
                VarianceThreshold(), 
                StandardScaler(), 
                PCA(svd_solver="full", n_components=0.95)
            ),
            classifier,
            classifier_params
        )
    )


def create_density_filter_default_pipeline(
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None   
) -> Pipeline:
    '''
    Creates the default pipeline in case of "density_filter" preprocessing.
    Allows to control whether to add ot not the classifier as last step.
    '''
    return make_pipeline(
        *add_classifier_head_to_steps(
            (
                VarianceThreshold(), 
                DensityFeatureSelector(
                    n_target_cols=500, 
                    strategy=density_feature_selector_strategy,
                    error_on_empty=True
                )
            ),
            classifier,
            classifier_params
        )
    )


def add_classifier_head_to_steps(
    steps: tuple,
    classifier: Classifier | None, 
    classifier_params: dict | None
) -> list:
    '''
    Add the classifier to the steps if not None.
    If it is None return steps.
    '''
    if classifier is not None:
        steps = [step for step in steps]
        steps.append(classifier(**classifier_params))
    return steps



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
 ) -> Classifier | Pipeline:
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
            The dict keys must be already adapted to the pipeline if any.

        return_fit_time (bool, optional):
            Whether tp return the fit time also along the fitted clf_or_pipe.
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

        # since we pop the classifier from the pipeline we must 
        # remove the classifier name from the fit_kwargs keys
        fit_classifier_kwargs = remove_string_from_params(
            params_dict=fit_classifier_kwargs, 
            string=f"{clf_or_pipe.steps[-1][0]}__"
        )

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
