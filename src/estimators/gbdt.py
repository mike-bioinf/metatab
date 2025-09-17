from __future__ import annotations

import pandas as pd
from copy import deepcopy
from typing import Literal, Callable, TYPE_CHECKING
from estimators.abstract_estimator import AbstractBaseEstimator
from estimators.utils import create_default_pipeline, fit_with_early_stop_on_validation_set
from hp_search.searchcv import SearchCV
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from estimators.constants import Classifier



class GBDTBaseEstimator(AbstractBaseEstimator):
    '''
    Base class for GBDT estimators. 
    Centralizes and standardizes the interface for the GBDT estimators.
    In detail it manages/abstracts the creation of the gbdt estimator and the fitting process.
    The fitting process is delegated to "SearchCV", "fit_with_early_stop_on_validation_set"
    and estimators "fit" methods depending on whether tuning and early stop are enabled. 

    Parameters:
        classifier_cls (Classifier): Classifier class.

        n_threads_parameter (str): 
            Name of the parameter controlling the 
            number of threads to use to fit the estimator.

        early_stopping (bool, optional):    
            Whether to fit using early stop on a validation set.

        callbacks_on_fixed_params (list[Callable[[dict, pd.Series, bool], dict]] | None, optional):
            List of functions to apply to the fixed/default params before fitting.
            They are applied sequentially following the list order.
            The output of the first is passed in input to the second and so on.
            They must share the same signature (params, y, do_copy) which is not checked in code.
            If None nothing is done.
        
        eval_set_parameter (str, optional):
            Name of the eval_set parameter, i.e. the
            parameter accepting the validation sets.
            Ignored when "early_stopping" is False. 

        validation_set_size (float, optional): 
            The size of the validation set. 
            Ignored when "early_stopping" is False.

        fit_classifier_kwargs (None | dict, optional):
            A dict unpackaged in the classifier fit calls.
            If None an empty dict is created.
            Useful to pass fit-level implementation-specific args.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_threads: int,
        early_stopping_rounds: int,
        tune_configuration: None | dict,
        *,
        classifier_cls: Classifier,
        n_threads_parameter: str,
        callbacks_on_fixed_params: list[Callable[[dict, pd.Series, bool], dict]] | None = None, 
        early_stopping: bool = False,
        eval_set_parameter: str = "eval_set",
        validation_set_size: float = 0.3,
        fit_classifier_kwargs: None | dict = None
    ):
        super().__init__(preprocessing, seed, n_threads, early_stopping_rounds, tune_configuration)
        self.classifier_cls = classifier_cls
        self.callbacks_on_fixed_params = callbacks_on_fixed_params
        self.n_threads_parameter = n_threads_parameter
        self.early_stopping = early_stopping
        self.eval_set_parameter = eval_set_parameter
        self.validation_set_size = validation_set_size
        self.fit_classifier_kwargs = fit_classifier_kwargs if fit_classifier_kwargs else {}


    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBDTBaseEstimator":
        # we are assuming that all concrete classes implementations follow the 
        # "random_state" and "early_stopping_rounds" parameter name convention.
        fixed_params = super().update_fixed_params(
            up_seed=True, 
            up_n_threads=True, 
            up_early_stopping_rounds=self.early_stopping,
            key_n_threads=self.n_threads_parameter,
            copy=True 
        )

        fixed_params = self._apply_callbacks_on_fixed_params(fixed_params, y)        
        pipe = self._create_pipeline(fixed_params)
        fit_classifier_kwargs = self._adjust_fit_kwargs_keys(pipe)

        if self.tune_configuration:
            self.estimator_ = SearchCV(
                clf_or_pipe=pipe,
                algo=self.tune_configuration["algo"],
                params_distributions=self.tune_configuration["params_distributions"],
                random_state_parameter="random_state",
                n_iter=self.tune_configuration["n_iter"],
                n_cv_repeats=self.tune_configuration["n_repeats"],
                n_cv_splits=self.tune_configuration["n_splits"],
                seed=self.seed,
                metric_to_minimize="logloss",
                fit_classifier_kwargs=fit_classifier_kwargs,
                early_stop_on_validation_set=self.early_stopping,
                validation_set_size=self.validation_set_size,
                eval_set_parameter=self.eval_set_parameter
            )
            self.estimator_.fit(X, y)

        elif self.early_stopping:
            self.estimator_ = fit_with_early_stop_on_validation_set(
                clf_or_pipe=pipe,
                X=X,
                y=y,
                seed=self.seed,
                validation_set_size=self.validation_set_size,
                eval_set_parameter=self.eval_set_parameter,
                fit_classifier_kwargs=fit_classifier_kwargs
            )

        else:
            self.estimator_ = pipe.fit(X, y, **fit_classifier_kwargs)

        return self


    def _create_pipeline(self, fixed_params: dict) -> Pipeline:
        return create_default_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy="oversample",
            classifier=self.classifier_cls,
            classifier_params=fixed_params
        )


    def _apply_callbacks_on_fixed_params(
        self, 
        params: dict, 
        y: pd.Series, 
        copy: bool = False
    ) -> dict:
        if self.callbacks_on_fixed_params:
            for cb in self.callbacks_on_fixed_params:
                params = cb(params, y, copy)
        return params


    def _adjust_fit_kwargs_keys(
        self, 
        clf_or_pipe: Classifier | Pipeline, 
    ) -> dict:
        '''
        Adjust the fit kwargs keys according to "clf_or_pipe" argument.
        Returns always a new dict.
        '''
        if isinstance(clf_or_pipe, Pipeline):
            name_classifier = clf_or_pipe.steps[-1][0]
            return {f"{name_classifier}__{k}":v for k, v in self.fit_classifier_kwargs.items()}
        else:
            return deepcopy(self.fit_classifier_kwargs)            



def adjust_objective_logloss_and_num_classes(
    params: dict, 
    y: pd.Series,
    framework: Literal["catboost", "xgboost", "lightgbm"],
    copy: bool,
) -> dict:
    '''
    GBDT classification framework implementations need to dinamically adjust the objective loss 
    and number of input classes depending on the classification scenario (binary or multi).
    In addiction different framework encode with different names the same losses functions.
    The function works on the input dict of params or a deepcopy of it (copy parameter).
    Returns the updated dict of params.
    '''
    params = deepcopy(params) if copy else params
    n_classes = y.unique().size
    
    if framework == "catboost":
        loss_parameter = "loss_function"
        binary_logloss = "Logloss"
        multi_logloss = "MultiClass"
        n_classes_parameter = "classes_count"
    elif framework == "xgboost":
        loss_parameter = "objective"
        binary_logloss = "binary:logistic"
        multi_logloss = "multi:softprob"
        n_classes_parameter = "num_class"
    elif framework == "lightgbm":
        loss_parameter = "objective"
        binary_logloss = "binary"
        multi_logloss = "multiclass"
        n_classes_parameter = "num_class"
    else:
        raise ValueError("Unsupported GBDT framework.")
    
    if n_classes == 2:
        params[loss_parameter] = binary_logloss
    else:
        params[loss_parameter] = multi_logloss
        params[n_classes_parameter] = n_classes

    return params



def adjust_es_logloss_metric(
    params: dict, 
    y: pd.Series, 
    framework: Literal["catboost", "xgboost", "lightgbm"], 
    copy: bool
) -> dict:
    '''
    The GBDT classification implementations differentiate between 
    binary and multi logloss as early stopping metric.
    The function adjust the evaluation logloss metric marked as "logloss_to_adjust".
    The function works on the input dict of params or a deepcopy of it (copy parameter).
    Returns the updated dict of params.
    '''
    params = deepcopy(params) if copy else params
    n_classes = y.unique().size

    if framework == "catboost":
        metric_parameter = "eval_metric"
        binary_logloss = "Logloss"
        multi_logloss = "MultiClass"
    elif framework == "xgboost":
        metric_parameter = "eval_metric"
        binary_logloss = "logloss"
        multi_logloss = "mlogloss"
    elif framework == "lightgbm":
        metric_parameter = "metric"
        binary_logloss = "binary_logloss"
        multi_logloss = "multi_logloss"
    else:
        raise ValueError("Unsupported GBDT framework.")
    
    # return the params dict when the early stop metric is not the adjustable logloss
    if params[metric_parameter] != "logloss_to_adjust":
        return params
    
    if n_classes == 2:
        params[metric_parameter] = binary_logloss
    else:
        params[metric_parameter] = multi_logloss
    
    return params