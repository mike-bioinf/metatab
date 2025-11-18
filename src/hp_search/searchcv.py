import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Literal
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from hyperopt import STATUS_OK, STATUS_FAIL, tpe, rand, fmin, space_eval
from hyperopt.pyll.stochastic import sample
from estimators.utils.types import Classifier, PreprocessingStrategy, TunableEstimatorType
from estimators.utils.fit import fit_with_early_stop_on_validation_set
from estimators.params import HPS_MIXED_TYPED
from hp_search.point_corrector import PointCorrector
from metalearning.metafeatures import CustomMFE
from metalearning.load import query_surrogate_framework
from metalearning.acquisition_funcs import compute_upper_confidence_bound
from metalearning.surrogate_worker import SurrogateWorker
from metalearning.sampler import HyperoptRandomSampler
from metalearning.generator import MetadataGenerator
from metatab_utils.general import add_broadcasted_objects_as_column
from hp_search.cv import CrossValidator
from hp_search.types import MetaAlgo, MetaStrategy, MetaStrategyParams

from hp_search.utils import (
    ConfigSearchCV, 
    set_params_into_clf,
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams
)



class SearchCV:
    '''
    Class that implements HPs optimization via random search or
    tpe methods with (repeated) cross-validation.

    Some key features:
    - Allows a meta-learning informed search via surrogate models using the "meta" algo.
    - Allows early stop on a validation set at fit time. This functionality is possible 
    only when the classifier implements an "eval_set-like" interface.
    - Allows to optionally refit the classifier/pipeline with the best hyperparameters.
    - Exposes the "predict_proba" method of the refitted object.
    - The search is not parallelized even when the "random" algo is selected. 
    

    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Classifier or Pipeline object with a classifier as head, 
            which hps have to be optimized.
        
        type_estimator (TunableEstimatorType):
            String estimator type. Info needed in meta-optimization (`meta` algo).
            
        type_clf_or_pipe_preprocessing (PreprocessingStrategy | None):
            Type of preprocessing used for the clf_or_pipe object.
            Needed only by the `meta` algo to propose candidate points.
        
        algo (Literal["random", "tpe", "meta"]): 
            Type of searching algorithm to use.
        
        params_distributions (dict): Search space.

        n_iter (int): Number of search iterations.
        
        n_cv_folds (int): Number of cv folds.

        n_cv_repeats (int): Number of cv repeats.
        
        seed (int): 
            Seed for reproducibility.
            Used in the standard optimization routes to draw points (random, tpe), 
            in the determination of the validation set when early stop is enabled,
            and in the inner-cross validation procedure (indipendently of the tune algo).

        random_state_parameter (str | None):
            Name of the estimator random state parameter.
        
        metric_to_minimize (Literal["logloss"]):
            The metric to minimize in the search.

        early_stop_on_validation_set (bool):
            Whether to early stop on validation set(s).

        eval_set_parameter (str | None, optional):
            Name of the eval_set parameter, i.e. the parameter taking the 
            validation set(s) at fit level. Can be None.
            Ignored when "early_stop_on_validation_set" is False.
        
        validation_set_size (float, optional):
            The ratio of the early stop validation set.
            Inside cv this set is taken from the training portion.
            Ignored when "early_stop_on_validation_set" is False.

        fit_classifier_kwargs (None | dict, optional):
            A dict unpackaged in the classifier fit calls.
            If None (default) an empty dict is created.
            The dict keys must be already adapted to the pipeline if any.
        
        meta_surrogate_model (None | str | Path, optional):
            Surrogate model to use in the meta-optimization scenario.
            If str or Path, then the object pointed by the path is used as surrogate model.
            This must be a joblib serialized object.
            If None the "default" surrogate model acording to `type_estimator` is used instead.
            Ignored when `algo` is not "meta".

        meta_strategy (MetaStrategy, optional):
            Set the strategy used by the metalearning framework to select points.
            It has no effect when `algo` is different from "meta".
            In detail the following `SurrogateWorker` utilities are used:
            - "best": `propose_n_best`
            - "random_from_best": `propose_random_from_top`
            - "uniform_from_best": `propose_best_uniform`
            See the specific method for more details.

        meta_seed (int, optional):
            Seed used specifically to draw condidate points in the meta-optimization scenario.
            Importanlty the default value of 42 is the one used to generate the prior.
            Therefore using the default seed allow to draw and evaluate real-evaluated 
            points. It's therefore highly suggested to not change this value in most
            applications.

        raise_error_during_search (None | bool, optional):
            Whether to ignore the errors during the search.
            If None, the parameter is set via the global `ConfigSearchCV` configuration class.

        build_df_search (None | bool, optional):
            Whether to build the DataFrame with complete search information.
            If False, the required information is not stored.
            This step can be indeed memory and time consuming.
            If None, the parameter is set via the global `ConfigSearchCV` configuration class.
            Forced to False if `n_iter` equal 1. 

        refit_with_best_hps (None | bool, optional):
            Whether to refit the clf_or_pipe with the best hps from the search.
            If None, the parameter is set via the global `ConfigSearchCV` configuration class.

        save_realtime_df_search_filepath (None | str | Path, optional):
            Whether to save after each search iteration the df_search.
            Ignored if `build_df_search` is False.
            If None, the parameter is set via the global `ConfigSearchCV` configuration class.
            Forced to False if `n_iter` equal 1. 
       

    ## Attributes:

        best_params_ (dict):
            Best HPs configuration obtained from the tuning procedure.
        
        best_estimator_ (Classifier | Pipeline):
            Refitted classifier/pipeline with the best hps configuration coming from the search.
            Available only when "refit_with_best_hps" is True.

        df_search_ (pd.DataFrame):
            Dataframe with the search info (hps and loss) at cv-fold level.
            Does not contain info about the failed iterations.
            Keep in mind that the the completed iterations are numerically 
            sequentially labeled at the end of the search ("search_iter" column).
            This means that if point n2 in the search fails, then point n3 is reported as 2 in the df.
            The attribute is set only when "build_df_search" flag is True.
        
        search_losses_ (list[float]):
            List of the losses registered during the search.
            Contains np.nan for failed iterations.
            The search order is respected.

        refit_time_ (float):
            Time of refit on the best configuration in seconds.
            Available only when "refit_with_best_hps" is True.
    '''
    def __init__(
        self,
        *,
        clf_or_pipe: Classifier | Pipeline,
        type_estimator: TunableEstimatorType,
        type_clf_or_pipe_preprocessing: PreprocessingStrategy | None,
        algo: MetaAlgo,
        params_distributions: dict,
        n_iter: int,
        n_cv_folds: int,
        n_cv_repeats: int,
        seed: int,
        random_state_parameter: str,
        metric_to_minimize: Literal["logloss"],
        early_stop_on_validation_set: bool,
        eval_set_parameter: str | None = "eval_set",
        validation_set_size: float = 0.3,
        fit_classifier_kwargs: None | dict = None,
        meta_surrogate_model: None | str | Path = None,
        meta_strategy: MetaStrategy = "random_from_best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_seed: int = 42,
        raise_error_during_search: None | bool = None,
        build_df_search: None | bool = None,
        refit_with_best_hps: None | bool = None,
        save_realtime_df_search_filepath: None | str | Path = None
    ):
        self.clf_or_pipe=clf_or_pipe
        self.type_estimator=type_estimator
        self.type_clf_or_pipe_preprocessing=type_clf_or_pipe_preprocessing
        self.algo=algo
        self.params_distributions=params_distributions
        self.random_state_parameter=random_state_parameter
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_folds=n_cv_folds
        self.seed=seed
        self.metric_to_minimize=metric_to_minimize
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs if fit_classifier_kwargs else {}
        self.meta_surrogate_model=meta_surrogate_model
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_seed=meta_seed
        
        self.cross_validator=CrossValidator(
            clf_or_pipe=clf_or_pipe,
            clf_random_state_parameter=random_state_parameter,
            early_stop_on_validation_set=early_stop_on_validation_set,
            eval_set_parameter=eval_set_parameter,
            validation_set_size=validation_set_size,
            fit_classifier_kwargs=self.fit_classifier_kwargs, # here we must always pass a dict
            metric=metric_to_minimize,
            n_folds=n_cv_folds,
            n_repeats=n_cv_repeats,
            seed=seed
        )

        # controlled by ConfigSearchCV
        self.raise_error_during_search: bool = ConfigSearchCV.get_setting(raise_error_during_search, "raise_error_during_search")
        self.build_df_search: bool = ConfigSearchCV.get_setting(build_df_search, "build_df_search")        
        self.refit_with_best_hps: bool = ConfigSearchCV.get_setting(refit_with_best_hps, "refit_with_best_hps")
        self.save_realtime_df_search_filepath: None | str | Path = ConfigSearchCV.get_setting(
            save_realtime_df_search_filepath, 
            "save_realtime_df_search_filepath"
        )
        


    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SearchCV":
        '''Performs HPO. Returns the instance. '''
        self._X = X
        self._y = y

        self._point_corrector = PointCorrector()
        self._set_point_to_model_corrections()
        
        self._dfs_info_iter: list[pd.DataFrame] = []
        self.search_losses_: list[float] = []

        # with n_iter equal 1 we skip the point evaluation
        # since the single point is the best by definition.
        # Therefore we cannot build the df_search.
        if self.n_iter == 1:
            # we append nan since we do not evaluate the loss
            self.search_losses_.append(np.nan)
            self.build_df_search = False
            self.save_realtime_df_search_filepath = False

        if self.algo == "meta":
            if self.type_clf_or_pipe_preprocessing is None:
                raise ValueError(
                    "'type_clf_or_pipe_preprocessing' cannot be None with 'meta' algo."
                )
            self._check_meta_strategy_params()
            self._fit_with_meta_points()
        
        elif self.algo in ["random", "tpe"]:
            self._fit_with_standard_algo()

        else:
            raise ValueError("Unsupported optimization algorithm.")
                
        # do not build the dataframe again when is build in realtime
        if self.build_df_search and not self.save_realtime_df_search_filepath:
            self.df_search_ = self._build_df_search()

        if self.refit_with_best_hps:
            best_estimator = deepcopy(self.clf_or_pipe)
            set_params_into_clf(best_estimator, self.best_params_)   
            
            if self.early_stop_on_validation_set:
                self.best_estimator_, self.refit_time_ = fit_with_early_stop_on_validation_set(
                    clf_or_pipe=best_estimator,
                    X=X,
                    y=y,
                    seed=self.seed,
                    validation_set_size=self.validation_set_size,
                    eval_set_parameter=self.eval_set_parameter,
                    fit_classifier_kwargs=self.fit_classifier_kwargs,
                    return_fit_time=True
                )
            else:
                start_refit_time = time.time()
                self.best_estimator_ = best_estimator.fit(X, y, **self.fit_classifier_kwargs)
                self.refit_time_ = time.time() - start_refit_time
            
        return self



    def _fit_with_meta_points(self) -> None:
        '''
        Optimize using the meta-inferred points only.        
        Set the `best_params_` attribute.
        '''
        # use the input model or use the default
        surrogate_model = joblib.load(self.meta_surrogate_model) \
            if self.meta_surrogate_model \
            else query_surrogate_framework(self.type_estimator)

        # we currently use only this acquisition function
        acquisition_func = partial(
            compute_upper_confidence_bound,
            k="infer_low", # we use the more conservative approach since tuning gives only one point
            mean_direction="lower_is_better", # we currently optimize only the logloss
            n_points=self.n_iter
        )

        meta_generator = MetadataGenerator(
            sampler=HyperoptRandomSampler(),
            point_corrector=PointCorrector(),
            mfe=CustomMFE(),
        )

        surrogate_worker = SurrogateWorker(
            metadata_generator=meta_generator,
            surrogate_framework=surrogate_model,
            acquisition_func=acquisition_func
        )

        surrogate_worker.fit(
            X=self._X,
            y=self._y,
            hp_space=self.params_distributions,
            seed=self.meta_seed
        )

        # 1500 are the points used in our prior for all estimators (for now)
        # In general drawning too many points can reduce their divergence and therefore hurt performance
        # TODO: use "new" points also?
        n_candidate_points = max(1500, self.n_iter) \
            if self.meta_strategy_params is None \
            else self.meta_strategy_params.n_candidate_points

        _ = surrogate_worker.draw_candidates(
            n_candidate_points=n_candidate_points,
            # add the preprocessing to the meta-data
            mfe_extract_kwargs = {"add_features": {"preprocessing": self.type_clf_or_pipe_preprocessing}}
        )

        _ = surrogate_worker.evaluate_candidates()

        if self.meta_strategy == "best":
            points = surrogate_worker.propose_n_best(n_best=self.n_iter)
        
        elif self.meta_strategy == "random_from_best":
            # we use a ratio of 1 to 3 by default when possible, 
            # meaning we give "3 choices for point"
            top = min(self.n_iter * 3, n_candidate_points) \
                if self.meta_strategy_params is None \
                else self.meta_strategy_params.top
            
            # we use the "normal" seed of the instance to allow variability,
            # when not hardcoded in the supplied params
            propose_seed = self.seed if self.meta_strategy_params is None else self.meta_strategy_params.seed

            points = surrogate_worker.propose_random_from_top(
                n_proposed=self.n_iter,
                top=top,
                seed=propose_seed
            )
        
        elif self.meta_strategy == "uniform_from_best":
            # we use a step of 3 by default when possible
            if self.meta_strategy_params is None:
                step_size = 3 if (n_candidate_points / self.n_iter) > 3 else 1
            else:
                step_size = self.meta_strategy_params.step_size
            
            points = surrogate_worker.propose_uniform_from_top(
                n_steps=self.n_iter,
                step_size=step_size
            )
        
        else:
            raise ValueError(
                "meta_strategy must be one of: 'best', 'random_from_best', 'uniform_from_best'."
            )

        # we do not evaluate the single point since is the best by definition
        if len(points) == 1:
            self.best_params_ = self._point_corrector.correct_point(
                points[0],
                **self._point_to_model_corrections
            )
            return None

        for point in points:
            _ = self._fit_point(point, returns_type="simple")

        losses = np.array(self.search_losses_)

        if np.isnan(losses).all():
            raise ValueError("All search iterations have failed.")
        
        self.best_params_ = self._point_corrector.correct_point(
            points[np.nanargmin(losses)],
            **self._point_to_model_corrections
        )


    
    def _fit_with_standard_algo(self) -> None:
        '''
        Optimize HPs with the random or tpe algo.
        Set the `best_params_` attribute.
        '''
        # with n_iter to 1 the sampling is always random 
        # and the drawn point is the best by definition
        if self.n_iter == 1:
            self.best_params_ = self._point_corrector.correct_point(
                point=sample(self.params_distributions, np.random.default_rng(self.seed)),
                **self._point_to_model_corrections
            )
            return None

        if self.algo == "random":
            algo_fn = rand.suggest
        elif self.algo == "tpe":
            # we use hyperopt defaults
            algo_fn = partial(
                tpe.suggest,
                n_startup_jobs=20,  # number of random init points
                n_EI_candidates=24,  # number of candidate points from which select the most promising at each iteration
                gamma=0.25 # top fraction of hps-configurations to use as good
            )
        else:
            raise ValueError("Unsupported optimization algorithm.")

        fit_point_fn = partial(self._fit_point, returns_type="hyperopt")

        best = fmin(
            fn=fit_point_fn,
            space=self.params_distributions,
            algo=algo_fn,
            max_evals=self.n_iter,
            trials=None,
            rstate=np.random.default_rng(self.seed),
            verbose=False
        )

        # hyperopt tracks the uncorrected params
        self.best_params_ = self._point_corrector.correct_point(
            space_eval(self.params_distributions, best),
            **self._point_to_model_corrections
        )



    def _fit_point(
        self, 
        params: dict,
        returns_type: Literal["hyperopt", "simple"]
    ) -> dict | float:
        '''
        Fit using the input tune space point.

        Parameters:
            params (dict): 
                Dict of hps to use (tune space point). Must be not corrected.
            returns_type (Literal["hyperopt", "simple"]):
                Whether returns a hyperopt compatible result or a simpler one.
                In the first case the function returns a dict with hyperopt
                compatible info, in the second only the loss.
        '''
        try:
            params_to_model = self._point_corrector.correct_point(
                params, 
                **self._point_to_model_corrections
            )

            loss, df_cv_info = self.cross_validator.fit(
                X=self._X, 
                y=self._y,
                params=params_to_model,
                agg="mean",
                collect_info=self.build_df_search
            )

            # The code should not fail a single time from here, 
            # but if it happens then we have external bug/problems that can be confused with 
            # failed optimization iteration (for example no space and then yes on disk).
            # We don't tolerate these errors since external to the optimization procedure.
            try:
                if self.build_df_search:
                    # this is to avoid edge behaviour when these dfs are concatenated with pd.concat: 
                    # - block value coercion.
                    # - block warning for concatenating full na columns.
                    df_cv_info = add_broadcasted_objects_as_column(
                        df=df_cv_info, 
                        dictionary=params, # we add the original not corrected point
                        convert_bool_to_str=False,
                        convert_none_to_str=False,
                        force_object_datatype=HPS_MIXED_TYPED,
                        check_matching_keys_cols=True,
                        check_non_builtin_types=True,
                        copy=False
                    )
                    
                    self._dfs_info_iter.append(df_cv_info)
                    
                    # here a bit inefficient since we rebuild multiple times
                    if self.save_realtime_df_search_filepath:
                        self.df_search_ = self._build_df_search()
                        self.df_search_.to_csv(self.save_realtime_df_search_filepath, sep="\t", index=False)
            
            except Exception as e:
                raise ValueError(
                    f"Encountered the following error during df_search building or saving process: {e}"
                )

            # this line must be placed after the df_search building code
            self.search_losses_.append(loss)

            if returns_type == "hyperopt":
                return {"loss": loss, "status": STATUS_OK}
            else:
                return loss
        
        except Exception as e:
            # we re-raise the error if not tollerated
            if self.raise_error_during_search:
                raise ValueError(
                    f"The following error is encountered during the search: {e}"
                )

            # we enforce "search_losses_" to be of length n_iter
            self.search_losses_.append(np.nan)
            
            if returns_type == "hyperopt":
                return {
                    "loss": np.nan, 
                    "status": STATUS_FAIL,
                    "exception": str(e)
                }
            else:
                return np.nan



    def _set_point_to_model_corrections(self) -> None:
        '''
        Defines and sets the arguments that the PointCollector must use to apply 
        the correction to the points before they are passed to the models for the fitting procedure.
        '''
        if self.type_estimator == "tabpfn":
            self._point_to_model_corrections = {
                "apply_hypeopt_corrections": True,
                "estimator": "tabpfn",
                "estimator_corrections": "all"
            }
        else:
            self._point_to_model_corrections = {
                "apply_hypeopt_corrections": True
            }



    def _build_df_search(self) -> pd.DataFrame:    
        # add search iter column and concat
        for i in range(len(self._dfs_info_iter)):
            self._dfs_info_iter[i]["search_iter"] = i
        return pd.concat(self._dfs_info_iter, axis=0, ignore_index=True)



    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)



    def _check_meta_strategy_params(self) -> None:
        '''Check that the meta strategy and related params are compatible'''
        if self.meta_strategy_params is None:
            return None
        
        if (
            self.meta_strategy == "best" and 
            not isinstance(self.meta_strategy_params, BestMetaStrategyParams)
        ):
            raise ValueError((
                "With 'best' meta_strategy a 'BestMetaStrategyParams'"
                " object is expected in meta_strategy_params."
            ))
        
        elif (
            self.meta_strategy == "random_from_best" and 
            not isinstance(self.meta_strategy_params, RandomFromBestMetaStrategyParams)
        ):
            raise ValueError((
                "With 'random_from_best' meta_strategy a 'RandomFromBestMetaStrategyParams'"
                " object is expected in meta_strategy_params."
            ))
        
        elif (
            self.meta_strategy == "uniform_from_best" and
            not isinstance(self.meta_strategy_params, UniformFromBestMetaStrategyParams)
        ):
            raise ValueError((
                "With 'uniform_from_best' meta_strategy a 'UniformFromBestMetaStrategyParams'"
                " object is expected in meta_strategy_params."
            ))