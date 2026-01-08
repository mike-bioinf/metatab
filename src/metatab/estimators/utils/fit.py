from __future__ import annotations

import warnings
import time
from typing import TYPE_CHECKING, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from metatab.estimators.utils.general import remove_prefix_from_params

if TYPE_CHECKING:
    from metatab.estimators.utils.types import Classifier
    from metatab.metatab_utils.types import XType, YType



def fit_with_early_stop_on_validation_set(
    *,
    pipe: Pipeline,
    X: XType,
    y: YType,
    seed: int,
    validation_set_size: float,
    eval_set_parameter: str,
    fit_classifier_kwargs: dict,
    return_fit_time: bool = False
 ) -> Pipeline | tuple[Pipeline, float]:
    '''
    Utility to fit an estimator with early stop on a validation set.
    The estimator must implement the early stop capability at its 
    fit interface, following a GBDT-like API ("eval_set-like" parameter).

    Parameters:
        pipe (Pipeline): 
            The pipeline to fit. Is assumed to end with a classifier.
        
        X (XType): Training feature space.
        
        y (YType): Training labels.
        
        seed (int): 
            Seed for reproducibility used ONLY in the train/val splitting.
        
        validation_set_size (float): 
            Ratio of training data to use as validation.
            Must be a number in (0, 1).
        
        eval_set_parameter (str): 
            Name of the parameter accepting the validation sets.

        fit_classifier_kwargs (dict):
            A dict unpackaged in the classifier fit calls.
            The dict keys can be either in classifier or pipeline "formats".

        return_fit_time (bool, optional):
            Whether to return the fit time along with the fitted pipe.
            If True returns a tuple [pipe, fit_time], otherwise pipe directly.

    Returns:
        Pipeline|tuple: 
        The fitted pipeline alone or in a tuple with the fit time.
    '''
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=validation_set_size,
        random_state=seed,
        stratify=y
    )

    # we fit the underlying classifier directly in every scenario  
    fit_classifier_kwargs = remove_prefix_from_params(
        params_dict=fit_classifier_kwargs, 
        string=f"{pipe.steps[-1][0]}__"
    )

    # we always consider the preprocessing in the fit time
    start_fit_time = time.time()

    if len(pipe) > 1:
        # we split the classifier from the preprocessing 
        # to avoid to repeat the preprocessing 2 times.
        # we fit in place the two components separately.
        clf: Classifier = pipe[-1]
        preprocessing_pipeline: Pipeline = pipe[:-1]
        X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
        X_val_transformed = preprocessing_pipeline.transform(X_val)

        clf.fit(
            X_train_transformed, y_train, 
            **{eval_set_parameter: [(X_val_transformed, y_val)]},
            **fit_classifier_kwargs
        )

    else:
        # we fit directly the classifier
        pipe[-1].fit(
            X_train, y_train,
            **{eval_set_parameter: [(X_val, y_val)]},
            **fit_classifier_kwargs
        )
    
    fit_time = time.time() - start_fit_time

    if return_fit_time:
        return [pipe, fit_time]
    else:
        return pipe
    


def set_params_into_clf(
    pipe: Pipeline, 
    params: dict[str, Any],
    set_tabpfn_inference_config: bool = True
) -> None:
    '''
    Set the classifier (pipeline head) parameters in place. 
    The method works with all classifiers, and with pipeline or classifier formatted params.
    The method overwrites the pre-existent parameters values for the ones specified in params.
    For tabpfn classifiers is possible to micro manage the setting of the `inference_config__` 
    marked parameters.
    '''
    clf: Classifier = pipe[-1]
    remove_prefix_from_params(params, string=f"{pipe.steps[-1][0]}__")

    if isinstance(clf, TabPFNClassifier):
        if "inference_config" in params.keys():
            raise KeyError(
                "The inference_config parameter cannot be handled explicity.",
                "Instead its keys must be passed as normal parameters marked with the 'inference_config__' prefix."
            )

        inference_config = {}
        cleaned_params = {}
        
        for k, v in params.items():
            if k.startswith("inference_config__"):
                inference_config[f"{k.removeprefix("inference_config__")}"] = v
            else:
                cleaned_params[k] = v

        if set_tabpfn_inference_config:
            if not inference_config:
                warnings.warn(
                    message=(
                        "Derived an empty inference_config dict."
                        " It will overwrite the classifier's existing inference_config."
                    ),
                    category=UserWarning
                )
            clf.set_params(inference_config=inference_config, **cleaned_params)
        else:
            if inference_config:
                warnings.warn(
                        message=(
                        "Derived a non-empty inference_config dict, but since "
                        "set_tabpfn_inference_config=False, it will be ignored."
                    ),
                    category=UserWarning
                )
            clf.set_params(**cleaned_params)
    
    else:
        clf.set_params(**params)