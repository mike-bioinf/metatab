from __future__ import annotations

import pickle
import pandas as pd
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING, Callable
from sklearn.pipeline import Pipeline
from metatab.preprocessing import create_classifier_pipeline
from metatab.estimators.utils.general import add_prefix_to_params_when_absent
from metatab.estimators.utils.fit import fit_with_early_stop_on_validation_set
from metatab.hp_search.searchcv import SearchCV
from metatab.ensemble.single import EnsembleEstimator
from metatab.preprocessing.utils import resolve_preprocessing_info
from metatab.metatab_utils.general import ensure_or_create, asdict_shallow

from metatab.metatab_utils.device import (
    check_device_estimator_combination, 
    check_cuda_is_available, 
    resolve_device, 
    check_device
)

if TYPE_CHECKING:
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.metatab_utils.types import XType, YType    
    from metatab.estimators.utils.types import Classifier, EstimatorType

    from metatab.estimators.core.configurations import (
        TuneConfiguration, 
        EarlyStopConfiguration, 
        EnsembleConfiguration
    )



class AbstractBaseEstimator(ABC):
    '''
    Abstract base class for estimators classes.

    Parameters:
        preprocessing (PreprocessingStrategy): Preprocessing strategy to use.
        seed (int): Seed for estimator reproducibility.
        n_threads (int): Number of CPU threads used to fit the estimator.
        device (Literal["cpu", "cuda", "auto"]): Where to run the estimator fitting process.
        early_stop_configuration (None | EarlyStopConfiguration, optional): Early stop configuration.
        tune_configuration (None | TuneConfiguration, optional): Tune configuration.
        ensemble_configuration (None | EnsembleConfiguration, optional): Ensemble configuration.
    
    ### Important Design Note:
        The estimators concrete classes must implement the `fixed_params` class attribute, 
        the dict with the fixed/default classifier parameters.
        !!!Attention: this is not checked/enforced by the code.
    '''
    if TYPE_CHECKING:
        fixed_params: dict

    def __init__(
        self, 
        preprocessing: PreprocessingStrategy,
        seed: int,
        n_threads: int,
        device: Literal["cpu", "cuda", "auto"],
        early_stop_configuration: None | EarlyStopConfiguration = None,
        tune_configuration: None | TuneConfiguration = None,
        ensemble_configuration: None | EnsembleConfiguration = None
    ):
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_threads=n_threads
        self.device=device
        self.early_stop_configuration=early_stop_configuration
        self.tune_configuration=tune_configuration
        self.ensemble_configuration=ensemble_configuration
        
    
    @abstractmethod
    def fit(X_train: pd.DataFrame, y_train: pd.Series):
        pass
    

    def save(self, filepath: str | Path, check_is_fitted: bool = False) -> None:
        '''
        Serielize the instance using pickle.
        Allows for a conditional check on the "fitted nature" of the estimator.
        '''
        if check_is_fitted and not hasattr(self, "estimator_"):
            raise ValueError("The estimator instance is not fitted (no 'estimator_' attibute).")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    def fit_estimator(
        self,
        *,
        X: XType,
        y: YType,
        classifier_cls: Classifier,
        type_estimator: EstimatorType,
        is_tuned: bool = False,
        is_ensembled: bool = False,
        is_early_stopped: bool = False,
        eval_set_parameter: str | None = "eval_set",
        early_stop_rounds_parameter: str | None = "early_stopping_rounds", 
        random_state_parameter: str | None = "random_state",
        n_threads_parameter: str | None = "n_jobs",
        device_parameter: str | None = None,
        callbacks_on_fixed_params: list[Callable[[dict, pd.Series, bool], dict]] | None = None,
        density_feature_selector_strategy: Literal["exact", "oversample", "undersample"] = "oversample",
        fit_classifier_kwargs: None | dict = None
    ) ->  Pipeline | SearchCV | EnsembleEstimator:
        '''
        Utility that abstracts the `fit` logic of concrete estimators.
        This function centralizes the repeated steps involved in preparing 
        and fitting the internal estimator involving:
        - Completing the `fixed_params` attribute of concrete estimators.
        - Creating the inner pipeline.
        - Fitting the inner estimator using the appropriate strategy.
        

        Parameters:
            X (XType): Data to fit.

            y (Ytype): Data labels to fit.
            
            classifier_cls (Classifier): Classifier class.
            
            type_estimator (EstimatorType): String estimator type.

            is_tuned (bool, optional):
                Whether the concrete estimator leverages HPs tuning.

            is_ensembled (bool, optional):
                Whether the concrete estimator leverages ensembling.

            is_early_stopped (bool, optional):
                Whether the concrete estimator leverages early stop on a validation set.

            early_stop_rounds_parameter (str | None, optional):
                Name of the classifier parameter accepting the number of early stop rounds info.
                None is used to signal that the classifier does not accept a early_stop_rounds-like
                parameter and therefore the info contained into `self.early_stop_configuration` is not used.
                In addition this info is ignored when `is_early_stopped` is False.

            eval_set_parameter (str | None, optional):
                Name of the classifier "eval_set-like" parameter, i.e. the parameter accepting the validation set(s).
                None is used to signal that the classifier does not accept a eval_set-like
                parameter and therefore the info contained into `self.early_stop_configuration` is not used.
                In addition this info is ignored when `is_early_stopped` is False.

            random_state_parameter (str | None, optional):
                Name of the classifier parameter accepting the random state info.
                None is used to signal that the classifier does not accept a random_state-like 
                parameter and therefore the `self.seed` info is unused.

            n_threads_parameter (str | None, optional): 
                Name of the classifier parameter accepting the number of threads info to use in fit.
                None is used to signal that the classifier does not accept a n_threads-like 
                parameter and therefore the `self.n_threads` info is unused.

            device_parameter (str | None, optional):
                Name of the classifier parameter acceting the device info.
                None is used to signal that the classifier does not have a device-like parameter.
                In this case the `self.device` info is unused.

            callbacks_on_fixed_params (list[Callable[[dict, pd.Series, bool], dict]] | None, optional):
                List of functions to apply to the fixed params before fitting.
                They are applied sequentially following the list order.
                The output of the first is passed in input to the second and so on.
                They must share the same signature (params, y, do_copy) (is not checked by the code).
                Pass an empty list or None to skip this functionality.

            density_feature_selector_strategy (Literal["exact", "oversample", "undersample"], optional):
                Strategy to follow when the concrete estimator `preprocessing` attribute is "density_filter".
            
            fit_classifier_kwargs (None | dict, optional):
                A dict unpackaged in the classifier fit calls.
                It can follow either the classifier or pipeline "format".
                Useful to pass fit-level implementation-specific args.
                If None an empty dict is used.

                
        Returns:
            Pipeline|SearchCV|EnsembleEstimator: 
            The fitted inner estimator.
        '''
        check_device(self.device)
        resolved_device = resolve_device(self.device, type_estimator)
        if resolved_device == "cuda": check_cuda_is_available()
        check_device_estimator_combination(resolved_device, type_estimator)

        self._check_tune_ensemble_flags(is_tuned, is_ensembled)
        self._check_early_stop_inputs(is_early_stopped, eval_set_parameter)

        params = deepcopy(self.fixed_params)

        if random_state_parameter: 
            params[random_state_parameter] = self.seed
        
        if n_threads_parameter: 
            params[n_threads_parameter] = self.n_threads

        if device_parameter:
            params[device_parameter] = resolved_device
        
        if is_early_stopped and early_stop_rounds_parameter: 
            params[early_stop_rounds_parameter] = self.early_stop_configuration.early_stop_rounds

        callbacks_on_fixed_params = ensure_or_create(callbacks_on_fixed_params, list)
        params = self._apply_callbacks_on_fixed_params(params, callbacks_on_fixed_params, y)
        resolved_preprocessing = resolve_preprocessing_info(self.preprocessing)

        pipe = create_classifier_pipeline(
            preprocessing=resolved_preprocessing,
            density_feature_selector_strategy=density_feature_selector_strategy,
            classifier=classifier_cls,
            classifier_params=params,
            type_estimator=type_estimator
        )

        fit_classifier_kwargs = add_prefix_to_params_when_absent(
            params_dict=ensure_or_create(fit_classifier_kwargs, dict),
            string=f"{pipe.steps[-1][0]}__"
        )
        
        if is_ensembled or is_tuned:
            val_set_size = self.early_stop_configuration.validation_set_size\
                if is_early_stopped\
                else 0.0

        if is_ensembled:
            # EnsembleEstimator address both early stop and normal scenarios   
            estimator = EnsembleEstimator(
                pipe=pipe,
                type_estimator=type_estimator,
                preprocessing=resolved_preprocessing,
                seed=self.seed,
                fit_classifier_kwargs=fit_classifier_kwargs,
                early_stop_on_validation_set=is_early_stopped,
                validation_set_size=val_set_size,
                eval_set_parameter=eval_set_parameter,
                **asdict_shallow(self.ensemble_configuration)
            )
            return estimator.fit(X, y)

        elif is_tuned:
            # SearchCV address both early stop and normal scenarios    
            estimator = SearchCV(
                pipe=pipe,
                type_estimator=type_estimator,
                preprocessing=resolved_preprocessing,
                random_state_parameter=random_state_parameter,
                seed=self.seed,
                metric_to_minimize="logloss",
                fit_classifier_kwargs=fit_classifier_kwargs,
                early_stop_on_validation_set=is_early_stopped,
                validation_set_size=val_set_size,
                eval_set_parameter=eval_set_parameter,
                **asdict_shallow(self.tune_configuration)
            )
            return estimator.fit(X, y)

        elif is_early_stopped:
            return fit_with_early_stop_on_validation_set(
                pipe=pipe,
                X=X,
                y=y,
                seed=self.seed,
                validation_set_size=self.early_stop_configuration.validation_set_size,
                eval_set_parameter=eval_set_parameter,
                fit_classifier_kwargs=fit_classifier_kwargs
            )

        else:
            return pipe.fit(X, y, **fit_classifier_kwargs)


    @staticmethod
    def _check_tune_ensemble_flags(is_tuned: bool, is_ensembled: bool) -> None:
        if is_tuned and is_ensembled:
            raise ValueError(
                "The estimator cannot be tuned and ensembled at the same time."
            )


    @staticmethod
    def _check_early_stop_inputs(is_early_stopped: bool, eval_set_parameter: str | None) -> None:
        # We perform the check on "eval_set_parameter" only excluding the "early_stop_rounds_parameter"
        # since the second cannot be used even when early stopping (realmlp).
        # On the contraty the first is always used.
        if is_early_stopped and eval_set_parameter is None:
            raise ValueError(
                "'eval_set_parameter' cannot be None when 'is_early_stopped' is True."
            )


    @staticmethod
    def _apply_callbacks_on_fixed_params( 
        params: dict,
        callbacks: list[Callable[[dict, pd.Series, bool], dict]],
        y: pd.Series, 
        copy: bool = False
    ) -> dict:
        '''
        Apply the callbacks on the input params sequentially.
        Note that the callbacks must follow a specific signature.
        Returns the modified params (copy or old object).
        '''
        params = deepcopy(params) if copy else params
        for callback in callbacks:
            params = callback(params, y, False)
        return params