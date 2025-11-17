from __future__ import annotations

import pickle
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING, Callable
from sklearn.pipeline import Pipeline
from hp_search.searchcv import SearchCV
from metatab_utils.general import ensure_or_create
from estimators.preprocessing import create_classifier_pipeline
from estimators.utils.fit import fit_with_early_stop_on_validation_set

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.decomposition import PCA
    from preprocessing.density_selector import DensityFeatureSelector
    from estimators.utils.types import (
        Classifier, 
        PREPROCESSING_STRATEGIES, 
        ALL_ESTIMATOR_TYPE
    )




class AbstractBaseEstimator(ABC):
    '''
    Abstract base class for estimators classes.

    Parameters:
        preprocessing (PREPROCESSING_STRATEGIES): Preprocessing strategy to use.
        seed (int): Seed for estimator reproducibility.
        n_threads (int, optional): Number of CPU threads used to fit the estimator. 
        early_stop_configuration (None | dict, optional): Early stop configuration.
        tune_configuration (None | dict, optional): Tune configuration.
        ensemble_configuration (None | dict, optional): Ensemble configuration.
    
    ### Important Design Note:
        The estimators concrete classes must implement the `fixed_params` class attribute, 
        the dict with the fixed/default classifier parameters.
        !!!Attention: this is not checked/enforced by the code.
    '''
    if TYPE_CHECKING:
        fixed_params: dict

    def __init__(
        self, 
        preprocessing: PREPROCESSING_STRATEGIES,
        seed: int,
        n_threads: int,
        early_stop_configuration: None | dict = None,
        tune_configuration: None | dict = None,
        ensemble_configuration: None | dict = None
    ):
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_threads=n_threads
        self.early_stop_configuration=early_stop_configuration
        self.tune_configuration=tune_configuration
        self.ensemble_configuration=ensemble_configuration
        
    
    @abstractmethod
    def fit(X_train: pd.DataFrame, y_train: pd.Series):
        pass
    

    def fit_estimator(
        self,
        *,
        X: pd.DataFrame,  ###TODO: check if all works with numpy arrays --> CUSTOM MFE BREAKS, density_selection breaks
        y: pd.Series,    ### TODO. check if all works if numpy arrays --> CUSTOM MFE BREAKS, density_selection breaks
        classifier_cls: Classifier,
        type_estimator: ALL_ESTIMATOR_TYPE,
        is_tuned: bool,
        is_early_stopped: bool,
        eval_set_parameter: str | None = "eval_set",
        early_stop_rounds_parameter: str | None = "early_stopping_rounds", 
        random_state_parameter: str = "random_state",
        n_threads_parameter: str | None = "n_jobs",
        callbacks_on_fixed_params: list[Callable[[dict, pd.Series, bool], dict]] | None = None,
        density_feature_selector_strategy: Literal["exact", "oversample", "undersample"] = "oversample",
        fit_classifier_kwargs: None | dict = None
    ) -> Classifier | Pipeline | SearchCV:
        '''
        Utility that abstracts the `fit` logic of concrete estimators.

        This function centralizes the repeated steps involved in preparing 
        and fitting the internal estimator involving:
        - Completing the `fixed_params` attribute of concrete estimators.
        - Creating the inner classifier or pipeline.
        - Fitting the inner estimator using the appropriate strategy.

     
        Parameters:
            X (pd.DataFrame): Data to fit.

            y (pd.Series): Data labels to fit.
            
            classifier_cls (Classifier): Classifier class.
            
            type_estimator (ALL_ESTIMATOR_TYPE): String estimator type.

            is_tuned (bool):
                Whether the concrete estimator leverages HPs tuning.

            is_early_stopped (bool):
                Whether the concrete estimator leverages early stop on a validation set.

            early_stop_rounds_parameter (str | None, optional):
                Name of the classifier parameter accepting the number of early stop rounds info.
                None is used to signal that the classifier does not accept a early_stop_rounds-like
                parameter and therefore the info contained into `self.early_stop_configuration` is not used.
                In addition this info is ignored when `is_early_stopped` is False.

            eval_set_parameter (str | None, optional):
                Name of the classifier "eval_set-like" parameter, 
                i.e. the parameter accepting the validation set(s).
                None is used to signal that the classifier does not accept a eval_set-like
                parameter and therefore the info contained into `self.early_stop_configuration` is not used.
                In addition this info is ignored when `is_early_stopped` is False.

            random_state_parameter (str, optional):
                Name of the classifier parameter accepting the random state info.
                We expect every classifier to have it so cannot be ignored.

            n_threads_parameter (str | None, optional): 
                Name of the classifier parameter accepting the number of threads info to use in fit.
                None is used to signal that the classifier does not accept a n_threads-like 
                parameter and therefore the `self.n_threads` info is not used.

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
                Must follow the "classifier" format (no pipeline format).
                Useful to pass fit-level implementation-specific args.
                If None an empty dict is used.

                
        Returns:
            Classifier|Pipeline|SearchCV: The fitted inner estimator.
        '''
        self._check_fit_early_stop_inputs(
            is_early_stopped,
            early_stop_rounds_parameter,
            eval_set_parameter
        )

        params = self._update_fixed_params(
            up_seed=True, 
            up_n_threads=False if n_threads_parameter is None else True,
            up_early_stop_rounds=is_early_stopped,
            key_seed=random_state_parameter,
            key_n_threads=n_threads_parameter,
            key_early_stop_rounds=early_stop_rounds_parameter,
            copy=True
        )

        callbacks_on_fixed_params = ensure_or_create(callbacks_on_fixed_params, list)
        params = self._apply_callbacks_on_fixed_params(params, callbacks_on_fixed_params, y)

        clf_or_pipe = create_classifier_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy=density_feature_selector_strategy,
            classifier=classifier_cls,
            classifier_params=params,
            type_estimator=type_estimator
        )

        fit_classifier_kwargs = self._adjust_fit_kwargs_keys_to_clf_or_pipe(
            fit_classifier_kwargs=ensure_or_create(fit_classifier_kwargs, dict),
            clf_or_pipe=clf_or_pipe
        )

        if is_tuned:
            # searchcv address both early stop and normal scenarios
            val_set_size = self.early_stop_configuration["validation_set_size"]\
                if is_early_stopped\
                else 0.0
            
            estimator = SearchCV(
                clf_or_pipe=clf_or_pipe,
                type_estimator=type_estimator,
                type_clf_or_pipe_preprocessing=self.preprocessing,
                algo=self.tune_configuration["algo"],
                params_distributions=self.tune_configuration["params_distributions"],
                random_state_parameter=random_state_parameter,
                n_iter=self.tune_configuration["n_iter"],
                n_cv_repeats=self.tune_configuration["n_repeats"],
                n_cv_splits=self.tune_configuration["n_folds"],
                seed=self.seed,
                metric_to_minimize="logloss",
                fit_classifier_kwargs=fit_classifier_kwargs,
                early_stop_on_validation_set=is_early_stopped,
                validation_set_size=val_set_size,
                eval_set_parameter=eval_set_parameter,
                meta_surrogate_model=self.tune_configuration["meta_surrogate_model"],
                meta_strategy=self.tune_configuration["meta_strategy"],
                meta_strategy_params=self.tune_configuration["meta_strategy_params"]
            )
            return estimator.fit(X, y)

        elif is_early_stopped:
            return fit_with_early_stop_on_validation_set(
                clf_or_pipe=clf_or_pipe,
                X=X,
                y=y,
                seed=self.seed,
                validation_set_size=self.early_stop_configuration["validation_set_size"],
                eval_set_parameter=eval_set_parameter,
                fit_classifier_kwargs=fit_classifier_kwargs
            )

        else:
            return clf_or_pipe.fit(X, y, **fit_classifier_kwargs)


    @staticmethod
    def _check_fit_early_stop_inputs(
        is_early_stopped: bool,
        early_stop_rounds_parameter: str | None = "early_stopping_rounds", 
        eval_set_parameter: str | None = "eval_set",
    ):
        if is_early_stopped:
            if early_stop_rounds_parameter is None:
                raise ValueError(
                    "'early_stop_rounds_paramater' cannot be None when 'is_early_stopped' is True."
                )
            if eval_set_parameter is None:
                raise ValueError(
                    "'eval_set_parameter' cannot be None when 'is_early_stopped' is True."
                )


    def _update_fixed_params(
        self,
        *,
        up_seed: bool = False, 
        up_n_threads: bool = False,
        up_early_stop_rounds: bool = False, 
        key_seed: str = "random_state", 
        key_n_threads: str = "n_jobs",
        key_early_stop_rounds: str = "early_stopping_rounds",
        copy: bool = False
    ) -> dict:
        '''
        Update the fixed params dict or a deepcopy of it with the seed, 
        n_threads and early_stop_rounds info. Returns the updated dict.
        '''
        fixed_params = deepcopy(self.fixed_params) if copy else self.fixed_params
        if up_seed: fixed_params[key_seed] = self.seed
        if up_n_threads: fixed_params[key_n_threads] = self.n_threads
        if up_early_stop_rounds: 
            fixed_params[key_early_stop_rounds] = self.early_stop_configuration["early_stop_rounds"]
        return fixed_params


    @staticmethod
    def _adjust_fit_kwargs_keys_to_clf_or_pipe(
        fit_classifier_kwargs: dict,
        clf_or_pipe: Classifier | Pipeline
    ) -> dict:
        '''
        Adjust the fit kwargs keys according to "clf_or_pipe" argument.
        Returns always a new dict.
        '''
        if isinstance(clf_or_pipe, Pipeline):
            name_classifier = clf_or_pipe.steps[-1][0]
            return {f"{name_classifier}__{k}":v for k, v in fit_classifier_kwargs.items()}
        else:
            return deepcopy(fit_classifier_kwargs)    


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


    def save(self, filepath: str | Path, check_is_fitted = False) -> None:
        '''
        Serielize the instance using pickle.
        Allows for a conditional check on the "fitted nature" of the estimator.
        '''
        if check_is_fitted and not hasattr(self, "estimator_"):
            raise ValueError("The estimator instance is not fitted (no 'estimator_' attibute).")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    

    def collect_fit_preprocessing_info(
        self, 
        clf_or_pipe: Classifier | Pipeline,
        return_on_classifier: Literal["empty_dict", "error"] = "empty_dict"
    ) -> dict:
        '''
        Returns the preprocessing info from a fitted pipeline.
        If a classifier is passed in input then it returns an empty dict.
        '''
        if not isinstance(clf_or_pipe, Pipeline):
            if return_on_classifier == "empty_dict":
                return {}
            elif return_on_classifier == "error":
                raise ValueError("Classifier in input. No preprocessing is done.")
            else:
                raise ValueError("Unsupported value for 'return_on_classifier' parameter.")
        
        # from here we deal with a pipeline
        if self.preprocessing == "pca":
            return self._collect_from_pca_preprocessing(clf_or_pipe)
        elif self.preprocessing == "density_filter":
            return self._collect_from_density_preprocessing(clf_or_pipe)
        elif self.preprocessing in ["base", "no"]:
            return {}
        else:
            raise ValueError("Unrecognized preprocessing.")

    
    @staticmethod
    def _collect_from_pca_preprocessing(pipeline: Pipeline) -> dict:
        '''Collect the pca related learned info'''
        pca: PCA = pipeline.named_steps["pca"]
        # we wrap the container objects to avoid errors 
        # in the building process of the prediction dataframe object
        return {
            "n_pca_components": pca.n_components_,
            "explained_variance_ratio": [pca.explained_variance_ratio_],
            "total_explained_variance_ratio": pca.explained_variance_ratio_.sum()
        }
    

    @staticmethod
    def _collect_from_density_preprocessing(pipeline: Pipeline) -> dict:
        '''Collect the density related learned info'''
        density_selector: DensityFeatureSelector = pipeline.named_steps["densityfeatureselector"]
        return {
            "density_selection_strategy": density_selector.strategy_,
            "n_target_features": density_selector.n_target_features_,
            "minimum_selected_density_score": density_selector.minimum_density_score_
        }