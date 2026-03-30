from __future__ import annotations

import time
import warnings
import numpy as np
from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.api.metaconfig import MetaConfig
from metatab.classifiers.registry import get_classifier_specs_from_registry
from metatab.search.search import OptunaSearch
from metatab.search.score import PipelineConfigurationCVScorer
from metatab.utils.pipeline import build_pipeline
from metatab.utils.core import fit_using_validation_set

from metatab.utils.api import ( 
    encode_y, 
    handle_device, 
    check_validation_set, 
    check_validation_set_classifier_combination
)

if TYPE_CHECKING:
    from metatab.preprocessing import PreprocessingStrategy
    from metatab.utils.types import XType, YType
    from metatab.utils.types import TunableClassifierType



class TuneClassifier(ClassifierMixin, BaseEstimator):
    '''
    Tune a classifier using a cross-validation strategy.

    Parameters:
        type_classifier (TunableClassifierType):
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
            Stored in `df_search_` if True. 
            Adds overhead and increases memory consumption.
            Note: when `n_iter` is 1 no optimization is done,
            since the single drawn point is considered the best by definition.
            Therefore the `df_search_` attribute is not created.

        raise_error_during_search (bool, optional):
            Whether to stop the search if an iteration fails.
            If True, failed iterations are skipped.
            When all iterations fail an error is always raised.

        refit_best_configuration (bool, optional):
            Whether to refit the best configuration found in the search.
            If False then is not possible to predict on new data.
            Useful when one cares about the search info only. 

        meta_config (None | MetaConfig, optional):
            Config class for the "meta" algorithm.
            Expands to the default configuration when None and `tune_algo` equal "meta".
            Must be None when `tune_algo` != "meta", otherwise an error is raised.


    ## Attributes:

        classes_ (np.ndarray): 
            The array of class labels learnt at fit time.
        
        best_params_ (dict):
            Best HPs coming from the search.
        
        df_search_ (pd.DataFrame): 
            Dataframe that provides a summary of the optimization process.
            Exists only when `build_df_search` is True and `n_iter` is greater than 1.
    '''
    def __init__(
        self,
        type_classifier: TunableClassifierType,
        tune_algo: Literal["tpe", "random", "meta"],
        n_iter: int = 1,
        n_cv_repeats: int = 1,
        n_cv_folds: int = 5, 
        preprocessing: PreprocessingStrategy | list[PreprocessingStrategy] = "zero_variance",
        time_limit: float = 10_000_000,
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        build_df_search: bool = True,
        raise_error_during_search:  bool = False,
        refit_best_configuration: bool = True,
        meta_config: None | MetaConfig = None,
        log: int = 20
    ):
        self.type_classifier=type_classifier
        self.tune_algo=tune_algo
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_folds=n_cv_folds
        self.preprocessing=preprocessing
        self.time_limit=time_limit
        self.seed=seed
        self.n_threads=n_threads
        self.device=device
        self.build_df_search=build_df_search
        self.raise_error_during_search=raise_error_during_search
        self.refit_best_configuration=refit_best_configuration
        self.meta_config=meta_config
        self.log=log
            

    def fit(
        self,
        X: XType,
        y: YType,
        validation_set_size: None | float = None
    ) -> "TuneClassifier":
        '''
        Tune on training data.
        
        Parameters:
            X (XType): Data to fit.
            
            y (Ytype): Data labels to fit.
            
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
            self.meta_config._check()

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

        if self.build_df_search and self.n_iter == 1:
            warnings.warn((
                "No search is done when 'n_iter' equal 1." 
                " The 'build_df_search' parameter is forced to False."
            ))
            build_df_search = False
        else:
            build_df_search = self.build_df_search

        search_scorer = PipelineConfigurationCVScorer(
            X=X,
            y=y,
            n_folds=self.n_cv_folds,
            n_repeats=self.n_cv_repeats,
            metric="logloss",
            metric_aggregation="mean",
            validation_set_size=validation_set_size,
            seed=self.seed,
            n_threads=self.n_threads,
            device=resolved_device,
            store_cv_info=build_df_search
        )

        if self.tune_algo == "meta":
            ### we should do a search on n_iter meta-confs
            ## MetaSearch
            ## SearchWithFixedConfigurations
            ## best_conf (obtained)
            pass
        else:
            search_object = OptunaSearch(
                classifier_spec=classifer_spec,
                preprocessing=self.preprocessing,
                optuna_sampler=self.tune_algo,
                scorer=search_scorer,
                n_trials=self.n_iter,
                direction_optimization="minimize", # we use the logloss as score 
                time_limit=self.time_limit,
                seed=self.seed,
                log=self.log,
                raise_error_during_search=self.raise_error_during_search
            )
            best_conf = search_object.get_configurations()[0]


        if self.refit_best_configuration:
            best_pipe = build_pipeline(
                **best_conf.asdict(), 
                classifier_seed=self.seed,
                classifier_device=self.device,
                classifier_nthreads=self.n_threads,
                y=y
            )
              
            if best_conf.classifier_spec.early_stop_on_validation_set:
                self.estimator_, self.refit_time_ = fit_using_validation_set(
                    pipe=best_pipe,
                    X=X,
                    y=y,
                    validation_set_size=validation_set_size,
                    seed=self.seed,
                    return_fit_time=True
                )
            else:
                start_refit_time = time.time()
                self.estimator_ = best_pipe.fit(X, y)
                self.refit_time_ = time.time() - start_refit_time

        if build_df_search: 
            self.df_search_ = search_scorer.build_df_cv()
        self.classes_ = label_encoder.classes_
        self.best_conf_ = best_conf
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
        return self.estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.
        
        Returns:
            np.ndarray: The class probabilities of the input samples.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)