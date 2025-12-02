'''
In this module we implement the base meta estimator for the estimators py classes.
'''

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from estimators.core.configurations import TuneConfiguration, EarlyStopConfiguration

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from estimators.estimators import Estimator
    from metalearning.types import MetaStrategy, MetaStrategyParams
    from preprocessing.types import PreprocessingStrategy
    from metatab_utils.types import XType, YType



class BaseMetaEstimator:
    '''
    The meta estimators are backed by a meta-learning framework that suggestes,
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
        We highly suggest to NOT preprocess the microbial profiles
        apart expressing them in the "relative" format (i.e rows summing to 1).
        This is because we automatically select the most appropiate preprocessing 
        scheme according to the classifier.
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
            - "best": The best `n_iter` points are selected.
            - "random_from_best": `n_iter` points are randomly selected from the top.
            The top is automatically set as `min(n_iter * 3, n_candidate_points)`.
            This value be customized in `meta_strategy_params`.
            - "uniform_from_best": `n_iter` points are selected starting from the best 
            with a fixed step size. The step size is automatically set as 
            `3 if (n_candidate_points / self.n_iter) > 3 else 1`. 
            This value can be customized via `meta_strategy_params`.
            
        meta_strategy_params (None | MetaStrategyParams, optional):
            It's possible to customize the number of candidate points to draw
            as well as some point-selection options.
            See the specific dataclasses object to more info.

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


    ## Attributes:

        classes_ (int): The classes labels.

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
        n_threads: int = 1
    ):
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


    def fit(
        self,
        X: XType, 
        y: YType,
        preprocessing: PreprocessingStrategy,
        concrete_estimator_cls: Estimator,
        tuning_params: dict,
        early_stop_configuration: None | EarlyStopConfiguration
    ) -> "BaseMetaEstimator":
        '''
        Fit the meta estimator.

        Parameters:
            X (XType): Train data.
            
            y (YType): Train labels.
            
            concrete_estimator_class (Estimator): 
                Estimator class to instantiate.
            
            tuning_params (dict): 
                Tune space. Must be compatible with the surrogate model.
            
            early_stop_configuration (None | EarlyStopConfiguration):
                Must be implemented by the early stopped concrete estimators.
        
        Returns:
            self
        '''
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
            refit_with_best_hps=True,
            save_realtime_df_search_filepath=None
        )

        estimator: Estimator = concrete_estimator_cls(
            preprocessing=preprocessing,
            seed=self.seed,
            n_threads=self.n_threads,
            tune_configuration=tune_configuration,
            early_stop_configuration=early_stop_configuration
        )

        self.estimator_ = estimator.fit(X, y)
        fit_info = self.estimator_.collect_sklearn_fit_info()

        for k, v in fit_info.items():
            setattr(self, k, v)
        
        if self.build_df_search:
            self.df_search_ = self.estimator_.estimator_.df_search_

        self.search_losses_ = self.estimator_.get_search_losses()
        self.best_point_ = self.estimator_.get_best_hps()
        ## TODO: COLLECT ALSO METAPOINTS (???) --> WE SHOULD SAVE THEM IN SEARCHCV AS ATTRIBUTES
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