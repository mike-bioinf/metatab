from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from tabpfn import TabPFNClassifier
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from estimators.constants import Classifier




class ConfigSearchCV:
    '''
    Class that holds the globally configurable settings for SearchCV instances.
    
    - raise_error_during_search (bool):
        Control whether to ignore the errors during the search.
        If True an error in the fitting process of one point  
        will determine the failure of the entire search.

    - refit_with_best_hps (bool): 
        Control whether SearchCV refit the estimator with the best hps.
    
    - build_df_search (bool): 
        Control whether SearchCV builds the df_search when fitted.
    
    - save_realtime_df_search_filepath (str | Path | None):
        If not None allow to save the df_search after each search iteration at the specified path. 
        Ignored when "build_df_search" is False.
    '''
    raise_error_during_search = False
    refit_with_best_hps = True
    build_df_search = False
    save_realtime_df_search_filepath = None
    _attrs = [
        "raise_error_during_search",
        "refit_with_best_hps", 
        "build_df_search", 
        "save_realtime_df_search_filepath"
    ]

    @classmethod
    def get_setting(cls, value: Any, attr: str) -> Any:
        '''
        Returns the configuration or input value for the specified attribute. 
        The fallback on the global setting happens when the input value is None.
        '''
        if attr not in cls._attrs:
            raise ValueError(f"attr must be one of {cls._attrs}")
        if value is None:
            return getattr(cls, attr)
        else:
            return value




def set_params_into_clf(
    clf_or_pipe: Classifier | Pipeline, 
    params: dict[str, Any],
    set_tabpfn_inference_config: bool = True
) -> None:
    '''
    Set the parameters into the classifier in place. 
    The method works with all type of classifiers and even when they head pipeline objects.
    Note that the method expects 'classified formatted' parameters.
    The method overwrites the pre-existent parameters values for the ones specified in params.
    For tabpfn classifiers is possible to micro manage the setting of the `inference_config__` 
    marked parameters.
    '''
    clf = clf_or_pipe[-1] if isinstance(clf_or_pipe, Pipeline) else clf_or_pipe
    
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
