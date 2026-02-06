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

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.estimators.estimators import TunedEstimator
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.metatab_utils.types import XType, YType



class MetaTuneBaseEstimator(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_iter: int = 1,
        n_cv_repeats: int = 1,
        n_cv_folds: int = 5, 
        meta_strategy: MetaStrategy = "best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_surrogate_model: None | Pipeline = None,
        build_df_search: bool = False,
        meta_seed: int = 42,
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto"
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
            n_iter (int): 
                Number of search iterations (and therefore points evaluation).
        
            n_cv_folds (int): 
                Number of cv folds of the inner cross validation for point performance evaluation.

            n_cv_repeats (int): 
                Number of cv repeats of the inner cross validation for point performance evaluation.
            
            meta_strategy (MetaStrategy, optional):
                Set the strategy used by the metalearning framework to select points.
                It has no effect when `algo` is not "meta".
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

            build_df_search (bool, optional):
                Whether to build the dataframe with the search information.
                If True then the dataframe is contained in the  `df_search_` attribute.
                If False this attribute is not-existent.

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

            seed (int, optional): 
                Seed controlling the randomness inherent to the estimator,
                to the validation sets determination when early stop is enabled,
                and to the inner cross validation splits.

            n_threads (int, optional):
                Number of threads used to parallelize the classifier fitting process.
                Note that the search is not parallelized to avoid processes-related errors.
                In general is better to parallelize only one between models and search.
    
            device (Literal["cpu", "cuda", "auto"], optional):
                Device where to fit the model(s). 
                Note that for some estimators cannot be run on "cuda" raising an error.
                If "auto" then it selects cuda if available AND the estimator requires GPU else cpu.


        ## Attributes:

            classes_ (np.ndarray): The array of class labels learnt at fit time.

            n_features_in_ (int): Number of features seen during fit.
            
            feature_names_in_ (np.ndarray):
                Names of features seen during fit.
                Exists only when fitted on a pandas DataFrame with string column index.
            
            search_losses_ (np.ndarray):
                Array with the logloss scores of the evaluated points.
                When `n_iter` equals one, the score is NaN since the single
                point is best by definition and therefore no evaluated.
                Presence of NaN when `n_iter` > 1 indicates failed iterations.
            
            best_point_ (dict):
                Best HPs coming from the search.
                
            df_search_ (pd.DataFrame): 
                Search/optimization info at inner cross validation level.
                Exists only when `build_df_search` is True.
        '''
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_folds=n_cv_folds
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_surrogate_model=meta_surrogate_model
        self.build_df_search=build_df_search
        self.meta_seed=meta_seed
        self.seed=seed
        self.n_threads=n_threads
        self.device=device


    def fit(
        self,
        X: XType, 
        y: YType,
        preprocessing: PreprocessingStrategy,
        concrete_estimator_cls: TunedEstimator,
        tuning_params: dict,
        early_stop_configuration: None | EarlyStopConfiguration
    ) -> "MetaTuneBaseEstimator":
        '''
        Fit the metatune estimator.

        Parameters:
            X (XType): Train data.
            
            y (YType): Train labels.
            
            preprocessing (PreprocessingStrategy):
                Preprocessing strategy to use.
            
            concrete_estimator_class (Estimator): 
                Tuned estimator class to instantiate.
            
            tuning_params (dict): 
                Tune space. Must be compatible with the surrogate model.
            
            early_stop_configuration (None | EarlyStopConfiguration):
                Must be implemented by the early stopped concrete estimators.
        
        Returns:
            self
        '''
        check_X_y(X, y, dtype=None, ensure_all_finite=False)

        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity

        tune_configuration = TuneConfiguration(
            algo="meta",
            n_iter=self.n_iter,
            n_cv_repeats=self.n_cv_repeats,
            n_cv_folds=self.n_cv_folds,
            params_distributions=tuning_params,
            meta_strategy=self.meta_strategy,
            meta_strategy_params=self.meta_strategy_params,
            meta_surrogate_model=self.meta_surrogate_model,
            build_df_search=self.build_df_search,
            raise_error_during_search=False,
            refit_with_best_hps=True
        )

        estimator: TunedEstimator = concrete_estimator_cls(
            preprocessing=preprocessing,
            seed=self.seed,
            n_threads=self.n_threads,
            device=self.device,
            tune_configuration=tune_configuration,
            early_stop_configuration=early_stop_configuration
        )

        self.estimator_ = estimator.fit(X, y)

        for k, v in learn_sklearn_features_attributes(X).items():
            setattr(self, k, v)

        self.classes_ = le.classes_
        
        if self.build_df_search:
            self.df_search_ = self.estimator_.estimator_.df_search_

        self.search_losses_ = self.estimator_.get_search_losses()
        self.best_point_ = self.estimator_.get_best_hps()
        ## TODO: collect metapoints (???) --> we should have them in searchcv as attribute
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