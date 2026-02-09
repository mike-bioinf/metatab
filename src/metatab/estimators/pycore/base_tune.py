from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Literal
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from metatab.estimators.utils.general import check_predict_features
from metatab.estimators.core.configurations import TuneConfiguration, EarlyStopConfiguration
from metatab.estimators.utils.general import learn_sklearn_features_attributes
from metatab.estimators.params.utils import pick_estimator_tune_space
from metatab.estimators.utils.general import check_meta_tuning_options

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.estimators.estimators import TunedEstimator
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.metatab_utils.types import XType, YType



class MetaTuneInitializer:
    def __init__(
        self,
        n_iter: int = 1,
        n_cv_repeats: int = 1,
        n_cv_folds: int = 5, 
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        raise_error_during_search:  bool = False,
        build_df_search: bool = True,
        meta_strategy: MetaStrategy = "best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_surrogate_model: None | Pipeline = None,
        meta_seed: int = 42,
        preprocessing: PreprocessingStrategy = "estimator_default",
        algo: Literal["meta"] = "meta",
        tune_space: Literal["default"] = "default"
    ):
        '''
        The meta tuned estimators are backed by a meta-learning framework that suggestes,
        evaluates and selects points drawn from pre-defined tuning spaces.
        In detail the process consists of the following phases:

        1. Candidate points derivation.
        The candidate points are drawn in a fixed number, customizable via `meta_strategy_params`,
        in a reproducible way thanks to the `meta_seed`. We do NOT recommend to change these 
        values, since changing them means obtaining points that do not belong
        to our prior. In other words keeping the default assures to evaluate points
        that have all been tested on prior data.

        2. Promisingness score evaluation.
        The promisingness of the candidate points is evaluated though a surrogate model,
        fitted on our prior optimization-data, and the UCB acquistion function.
        In particular the surrogate model uses also a pre-defined set of metafeatures
        computed on the input data to predict point quality.

        3. Point selection.
        The `n_iter` most promising points are then selected using the `meta_strategy`.

        4. Optimization.
        The selected points are then evaluated using a inner cv cross validation on 
        the input data. The point with the best performance is finally selected.
        The performance is automatically evaluated using the logloss metric.
        This cannot be changed.


        ### User Note:
            We highly suggest to NOT preprocess the microbial profiles apart expressing them in the 
            "relative" format (i.e rows summing to 1). This is because we automatically select the 
            most appropiate preprocessing scheme according to the classifier.
            In addition we learn the data metafeatures before preprocessing. 
            Therefore a custom pre-preprocessing can potentially hurt performance.

        
        Parameters:
            n_iter (int, optional):
                Number of search iterations (number of hyperparameter configurations evaluated).

            n_cv_folds (int, optional):
                Number of folds for the inner cross-validation used to evaluate each hyperparameter configuration.

            n_cv_repeats (int, optional):
                Number of times the inner cross-validation is repeated for each hyperparameter configuration.
            
            seed (int, optional):
                Random seed controlling classifier randomness and the inner cross-validation procedure.

            n_threads (int, optional):
                Number of threads used to parallelize classifier fitting. 
                The hyperparameter search itself is not parallelized to avoid process overload.
                In general is better to parallelize either model fitting or search, but not both.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device to fit the model(s) on.
                - "cpu" or "cuda" explicitly selects the device.
                - "auto" uses GPU if available and supported by the classifier; otherwise CPU.

            raise_error_during_search (bool, optional):
                Whether to stop the search if a configuration evaluation fails.
                If False, failing evaluations are skipped. 
                If all evaluations fail, an error is always raised.

            build_df_search (bool, optional):
                Whether to build a dataframe containing search results. 
                Stored in `df_search_` if True. Adds minor overhead.
                Note: when `n_iter` is 1 no optimization is done,
                since the single drawn point is considered the best by definition.
                Therefore the `df_search_` attribute is not created.
            
            meta_strategy (MetaStrategy, optional):
                Set the strategy used by the metalearning framework to select points.
                - "best": select the top-n configurations.
                - "random_from_best": random selection from the top.
                - "uniform_from_best": uniform step selection from the top.
                - "random_uniform_from_best": random selection within uniform intervals from the top.
                
            meta_strategy_params (None | MetaStrategyParams, optional):
                Meta strategy specifics in form of dataclass.
                If None the default specifics are applied.

            meta_surrogate_model (None | str | Path, optional):
                Surrogate model to use for point performance prediction.
                If str or Path, then the object pointed by the path is used as surrogate model.
                This must be a joblib serialized object.
                If None the "default" surrogate model is used.

            meta_seed (int, optional):
                Seed specifically and only used to draw candidate points.
                The default value of 42 is the one used to generate our prior.
                Using this value allows to evaluate points tested beforehand. 
                Therefore is highly suggested to not modify this value.
                Note: If the parameter is left at its default but the number of 
                candidate points (set via `meta_strategy_params`) differs from the 
                default of 1500, the following occurs.
                - If the number of candidate points is less than 1500, 
                a subset of the prior points is selected.
                - If the number exceeds 1500, "new" points are drawn in addition 
                to the prior points.

            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy to apply. 
                Is highly suggested to leave "estimator_default",
                since other options could affect performance negatively.
                See the user note above for details.

            algo (Literal["meta"]):
                Must be "meta". Can be ignored by users.
            
            tune_space (Literal["default"]):
                Must be "default". Can be ignored by users.

                
        ## Attributes:

            classes_ (np.ndarray): 
                The array of class labels learnt at fit time.

            n_features_in_ (int): 
                Number of features seen during fit.
            
            feature_names_in_ (np.ndarray):
                Names of features seen during fit.
                Exists only when fitted on a pandas DataFrame with strings as columns.
            
            search_losses_ (np.ndarray):
                Array with the logloss scores of the evaluated points.
                When `n_iter` equals one, the array contains a single NaN since the single
                point is the best by definition, and therefore no evaluation is done.
                Presence of NaNs in the array when `n_iter` > 1 indicate failed iterations.
            
            best_hps_ (dict):
                Best HPs coming from the search.
                
            df_search_ (pd.DataFrame): 
                Dataframe that provides a summary of the optimization process.
                Exists only when `build_df_search` is True and `n_iter` is greater than 1.
        '''
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_folds=n_cv_folds
        self.seed=seed
        self.n_threads=n_threads
        self.device=device
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_surrogate_model=meta_surrogate_model
        self.meta_seed=meta_seed
        self.build_df_search=build_df_search
        self.raise_error_during_search=raise_error_during_search
        self.preprocessing=preprocessing
        self.algo=algo
        self.tune_space=tune_space


class StandardTuneInitializer:
    def __init__(
        self,
        n_iter: int = 1,
        n_cv_repeats: int = 1,
        n_cv_folds: int = 5, 
        algo: Literal["random", "tpe"] = "tpe",
        tune_space: str = "default",
        preprocessing: PreprocessingStrategy = "estimator_default",
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        raise_error_during_search:  bool = False,
        build_df_search: bool = True
    ):
        '''
        Optimize classifier hyperparameters via inner cross-validation.

        Parameters:

            n_iter (int, optional):
                Number of search iterations (number of hyperparameter configurations evaluated).

            n_cv_folds (int, optional):
                Number of folds for the inner cross-validation used to evaluate each hyperparameter configuration.

            n_cv_repeats (int, optional):
                Number of times the inner cross-validation is repeated for each hyperparameter configuration.

            algo (Literal["random", "tpe"], optional):
                Optimization algorithm to use.
                - "random": purely random search.
                - "tpe": Tree-structured Parzen Estimator. Performs an initial random warm-up of 20 iterations,
                  so at least 30 iterations are recommended for effective optimization.

            tune_space (str, optional):
                Pre-defined tuning space to use. 
                Use strings like "c{integer}" to select a specific space. 
                For non-GBDT classifiers, only "default" or "c0" should be used. 
                The default is recommended, as this parameter may be deprecated in the future 
                and internal tests showed no major performance differences among alternative spaces.
                
            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy to apply.
                Custom preprocessing cannot currently be applied within the inner cross-validation procedure.

            seed (int, optional):
                Random seed controlling classifier randomness and the inner cross-validation procedure.

            n_threads (int, optional):
                Number of threads used to parallelize classifier fitting. 
                The hyperparameter search itself is not parallelized to avoid process overload.
                In general is better to parallelize either model fitting or search, but not both.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device to fit the model(s) on.
                - "cpu" or "cuda" explicitly selects the device.
                - "auto" uses GPU if available and supported by the classifier; otherwise CPU.

            raise_error_during_search (bool, optional):
                Whether to stop the search if a configuration evaluation fails.
                If False, failing evaluations are skipped. 
                If all evaluations fail, an error is always raised.

            build_df_search (bool, optional):
                Whether to build a dataframe containing search results. 
                Stored in `df_search_` if True. Adds minor overhead.
                Note: when `n_iter` is 1 no optimization is done,
                since the single drawn point is considered the best by definition.
                Therefore the `df_search_` attribute is not created.

        
        ## Attributes:

            classes_ (np.ndarray): 
                The array of class labels learnt at fit time.

            n_features_in_ (int): 
                Number of features seen during fit.
            
            feature_names_in_ (np.ndarray):
                Names of features seen during fit.
                Exists only when fitted on a pandas DataFrame with strings as columns.
            
            search_losses_ (np.ndarray):
                Array with the logloss scores of the evaluated points.
                When `n_iter` equals one, the array contains a single NaN since the single
                point is the best by definition, and therefore no evaluation is done.
                Presence of NaNs in the array when `n_iter` > 1 indicate failed iterations.
            
            best_hps_ (dict):
                Best HPs coming from the search.
                
            df_search_ (pd.DataFrame): 
                Dataframe that provides a summary of the optimization process.
                Exists only when `build_df_search` is True and `n_iter` is greater than 1.
        '''
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_folds=n_cv_folds
        self.algo=algo
        self.tune_space=tune_space
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_threads=n_threads
        self.device=device
        self.raise_error_during_search=raise_error_during_search
        self.build_df_search=build_df_search


class BaseTune(ClassifierMixin, BaseEstimator):
    def fit(
        self,
        X: XType,
        y: YType,
        early_stop_configuration: None | EarlyStopConfiguration = None
    ):
        check_X_y(X, y, dtype=None, ensure_all_finite=False)

        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity
        
        # preliminary check on enforce algo value.
        if self.algo != "meta" and self.enforce_meta_algo:
            raise ValueError("'algo' must be equal to 'meta'.")

        if self.algo in ["random", "tpe"]:
            meta_tune_parameters = {}
        
        elif self.algo == "meta":
            # only this check is necessary here
            check_meta_tuning_options(
                estimator=self.type_estimator,
                preprocessing=self.preprocessing,
                tune_space=self.tune_space
            )

            meta_tune_parameters = {
                "meta_strategy": self.meta_strategy,
                "meta_strategy_params": self.meta_strategy_params,
                "meta_surrogate_model": self.meta_surrogate_model,
                "meta_seed": self.meta_seed
            }

        else:
            raise ValueError(f"Unsupported algo: '{self.algo}'")
        
        tune_configuration = TuneConfiguration(
            algo=self.algo,
            n_iter=self.n_iter,
            n_cv_repeats=self.n_cv_repeats,
            n_cv_folds=self.n_cv_folds,
            params_distributions=pick_estimator_tune_space(self.type_estimator, self.tune_space),
            raise_error_during_search=self.raise_error_during_search,
            build_df_search=self.build_df_search,
            refit_with_best_hps=True,
            **meta_tune_parameters,
        )

        estimator: TunedEstimator = self.myclass(
            preprocessing=self.preprocessing,
            seed=self.seed,
            n_threads=self.n_threads,
            device=self.device,
            tune_configuration=tune_configuration,
            early_stop_configuration=early_stop_configuration
        )

        self.estimator_ = estimator.fit(X, y)

        if self.build_df_search and self.n_iter > 1:
            self.df_search_ = self.estimator_.estimator_.df_search_

        for k, v in learn_sklearn_features_attributes(X).items():
            setattr(self, k, v)
        
        self.classes_ = le.classes_
        self.search_losses_ = self.estimator_.get_search_losses()
        self.best_hps_ = self.estimator_.get_best_hps()
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
        check_predict_features(self, X)
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
        check_predict_features(self, X)
        return self.estimator_.predict_proba(X)