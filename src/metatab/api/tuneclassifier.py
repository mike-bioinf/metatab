from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.hp_search.searchcv import SearchCV
from metatab.utils.general import asdict_shallow, ensure_or_create
from metatab.api.metaconfig import MetaConfig
from metatab.classifiers.registry import get_classifier_specs_from_registry

from metatab.utils.api import (
    create_pipeline, 
    encode_y, 
    handle_device, 
    check_validation_set, 
    check_validation_set_classifier_combination
)

if TYPE_CHECKING:
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.utils.types import XType, YType
    from metatab.utils.types import TunableEstimatorType



class TuneClassifier(ClassifierMixin, BaseEstimator):
    '''
    Class to tune a classifier using a cross-validation strategy.

    Parameters:

        type_classifier (TunableEstimatorType):
            Classifier to optimize.

        tune_algo (Literal["random", "tpe", "meta"]):
            Optimization algorithm to use.
            - "random": random search.
            - "tpe": Tree-structured Parzen Estimator. 
            Performs an initial random warm-up of 20 iterations,
            so at least 30 iterations are recommended for effective optimization.
            - "meta": uses Metatab meta-framework to evaluate meta-informed configurations.

        n_iter (int, optional):
            Number of search iterations (number of hyperparameter configurations evaluated).

        n_cv_folds (int, optional):
            Number of folds for the inner cross-validation used to evaluate each hyperparameter configuration.

        n_cv_repeats (int, optional):
            Number of times the inner cross-validation is repeated for each hyperparameter configuration.

        preprocessing (PreprocessingStrategy, optional):
            Preprocessing strategy to apply.
            Custom preprocessing cannot currently be applied within the inner cross-validation procedure.

        seed (int, optional):
            Random seed controlling classifier randomness and the inner cross-validation procedure.

        n_threads (int, optional):
            Number of threads used to parallelize classifier fitting. 
            The search is not parallelized to avoid process overload issues.
            In general is better to parallelize either model fitting or search, but not both.

        device (Literal["cpu", "cuda", "auto"], optional):
            Device to fit the model(s) on.
            - "cpu" or "cuda" explicitly selects the device.
            - "auto" falls on "cuda" if available and supported by the classifier; otherwise "cpu".
        
        build_df_search (bool, optional):
            Whether to build a dataframe containing search info. 
            Stored in `df_search_` if True. Adds overhead.
            Note: when `n_iter` is 1 no optimization is done,
            since the single drawn point is considered the best by definition.
            Therefore the `df_search_` attribute is not created.

        raise_error_during_search (bool, optional):
            Whether to stop the search if an iteration fails.
            If False, failing iterations are skipped. 
            When all iterations fail an error is always raised.

        meta_config (None | MetaConfig, optional):
            Config class for the "meta" algorithm.
            Expands to the default configuration when None and `tune_algo` equal "meta".
            Must be None when `tune_algo` != "meta", otherwise an error is raised.


    ## Attributes:

        classes_ (np.ndarray): 
            The array of class labels learnt at fit time.
        
        best_params_ (dict):
            Best HPs coming from the search.

        search_losses_ (np.ndarray):
            Array with the logloss scores of the evaluated points.
            When `n_iter` equals one, the array contains a single NaN since the single
            point is the best by definition, and therefore no evaluation is done.
            Presence of NaNs in the array when `n_iter` > 1 indicate failed iterations.
        
        df_search_ (pd.DataFrame): 
            Dataframe that provides a summary of the optimization process.
            Exists only when `build_df_search` is True and `n_iter` is greater than 1.
    '''
    def __init__(
        self,
        type_classifier: TunableEstimatorType, ##refactor type
        tune_algo: Literal["tpe", "random", "meta"],
        n_iter: int = 1,
        n_cv_repeats: int = 1,
        n_cv_folds: int = 5, 
        preprocessing: PreprocessingStrategy = "estimator_default",
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        build_df_search: bool = True,
        raise_error_during_search:  bool = False,
        meta_config: None | MetaConfig = None
    ):
        self.type_classifier = type_classifier
        self.tune_algo = tune_algo
        self.n_iter = n_iter
        self.n_cv_repeats = n_cv_repeats
        self.n_cv_folds = n_cv_folds
        self.preprocessing = preprocessing
        self.seed = seed
        self.n_threads = n_threads
        self.device = device
        self.build_df_search = build_df_search
        self.raise_error_during_search = raise_error_during_search
        self.meta_config = meta_config
            

    def fit(
        self,
        X: XType,
        y: YType,
        validation_set_size: None | float = None
    ) -> "TuneClassifier":
        '''
        Tune the classifier.
        
        Parameters:
            X (XType): 
                Data to fit.
            
            y (Ytype): 
                Data labels to fit.
            
            validation_set_size (None | float, optional): 
                Size of the validation set.
                Must be provided only for classifiers that use the validation set in the fit process.
                An error is raised when it is provided and the classifier does not support it,
                or when it is not provided and the classifier needs it.
        
        Returns:
            self
        '''
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        classifer_spec = get_classifier_specs_from_registry(self.type_classifier)

        if self.tune_algo not in ["random", "tpe", "meta"]:
            raise ValueError(f"Unsupported tune_algo: '{self.tune_algo}'.")
        
        if self.tune_algo != "meta" and self.meta_config:
            raise ValueError(f"With 'tune_algo' != 'meta', 'meta_config' must be None.")
        
        if self.tune_algo == "meta" and self.meta_config is not None:
            self.meta_config.check()

        if self.tune_algo == "meta" and self.meta_config is None:
            self.meta_config = MetaConfig()

        check_validation_set(validation_set_size)
        
        check_validation_set_classifier_combination(
            validation_set=validation_set_size,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )

        resolved_device = handle_device(
            input_device=self.device,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )

        # execute configuration code needed for the optimization task
        classifer_spec.initialize_search_function()
        label_encoder, y = encode_y(X, y)
        
        resolved_preprocessing = classifer_spec.default_preprocessing \
            if self.preprocessing == "estimator_default" \
            else self.preprocessing
        
        pipe = create_pipeline(
            classifier_class=classifer_spec.classifier_class,
            classifier_params=classifer_spec.fixed_params,
            callbacks_on_classifier_params=classifer_spec.callbacks_on_params,
            y=y,
            preprocessing=resolved_preprocessing,
            classifier_random_state_parameter=classifer_spec.random_state_parameter,
            classifier_nthreads_paramater=classifer_spec.n_threads_parameter,
            classifier_device_parameter=classifer_spec.device_parameter,
            seed=self.seed,
            n_threads=self.n_threads,
            device=resolved_device
        )

        estimator = SearchCV(
            pipe=pipe,
            type_estimator=self.type_classifier, ##refactor name parameter here
            preprocessing=resolved_preprocessing,
            algo=self.tune_algo,
            sampler_function=classifer_spec.sampler_function,
            n_iter=self.n_iter,
            n_cv_folds=self.n_cv_folds,
            n_cv_repeats=self.n_cv_repeats,
            seed=self.seed,
            random_state_parameter=classifer_spec.random_state_parameter,
            metric_to_minimize="logloss",
            early_stop_on_validation_set=classifer_spec.early_stop_on_validation_set,
            validation_set_size=validation_set_size,
            raise_error_during_search=self.raise_error_during_search,
            build_df_search=self.build_df_search,
            params_as_object_columns_in_df_search=classifer_spec.params_as_object_columns_in_df_search,
            **ensure_or_create(asdict_shallow(self.meta_config), dict) ## REFACTOR: here pass directly the metaconfig
        )

        self.estimator_ = estimator.fit(X, y)
        if self.build_df_search and self.n_iter > 1:
            self.df_search_ = self.estimator_.df_search_
        self.classes_ = label_encoder.classes_
        self.search_losses_ = self.estimator_.search_losses_
        self.best_params_ = self.estimator_.best_params_ ## REFACTOR: changed attributes from best_hps_ (check for effect)
        return self


    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict class for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            np.ndarray: The predicted classes.
        '''
        check_is_fitted(self, "estimator_")
        check_is_fitted(self.estimator_, "best_estimator_")
        return self.estimator_.best_estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.
        
        Returns:
            np.ndarray: The class probabilities of the input samples.
        '''
        check_is_fitted(self, "estimator_")
        check_is_fitted(self.estimator_, "best_estimator_")
        return self.estimator_.best_estimator_.predict_proba(X)