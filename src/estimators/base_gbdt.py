from __future__ import annotations

import pandas as pd
from typing import Literal, Callable, override, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from estimators.abstract_estimator import AbstractBaseEstimator
from estimators.utils import create_default_pipeline, fit_with_early_stop_on_validation_set
from estimators.searchcv import SearchCV
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from estimators.types import Classifier



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
            They must share the same signature (not checked in code).
            If None nothing is done on the dict of params.
        
        eval_set_parameter (str, optional):
            Name of the eval_set parameter, i.e. the
            parameter accepting the validation sets.
            Ignored when "early_stopping" is False. 

        validation_set_size (float, optional): 
            The size of the validation set. 
            Ignored when "early_stopping" is False.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_threads: int,
        tune_configuration: None | dict,
        fixed_params: dict,
        *,
        classifier_cls: Classifier,
        n_threads_parameter: str,
        callbacks_on_fixed_params: list[Callable[[dict, pd.Series, bool], dict]] | None = None, 
        early_stopping: bool = False,
        eval_set_parameter: str = "eval_set",
        validation_set_size: float = 0.3
    ):
        super().__init__(preprocessing, seed, n_threads, tune_configuration, fixed_params)
        self.classifier_cls = classifier_cls
        self.callbacks_on_fixed_params = callbacks_on_fixed_params
        self.n_threads_parameter = n_threads_parameter
        self.early_stopping = early_stopping
        self.eval_set_parameter = eval_set_parameter
        self.validation_set_size = validation_set_size


    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBDTBaseEstimator":
        fixed_params = super().update_fixed_params(
            up_seed=True, 
            up_n_threads=True, 
            key_n_threads=self.n_threads_parameter,
            copy=True 
        )

        fixed_params = self._apply_callbacks_on_fixed_params(fixed_params, y)        
        pipe = self._create_pipeline(fixed_params)

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
                eval_set_parameter=self.eval_set_parameter
            )

        else:
            self.estimator_ = pipe.fit(X, y)

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


    @override
    def get_best_hps(self) -> dict | None:
        if self.tune_configuration:
            check_is_fitted(self, "estimator_")
            return self.estimator_.best_params_
        return None