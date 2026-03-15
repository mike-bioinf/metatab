from __future__ import annotations

import time
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Literal, TYPE_CHECKING, Callable
from sklearn.pipeline import Pipeline
from optuna.samplers import RandomSampler, TPESampler
from metatab.utils.core import fit_with_early_stop_on_validation_set, set_params_into_clf
from metatab.utils.exceptions import DFSearchBuildingError
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.load import query_surrogate_framework
from metatab.metalearning.acquisition_funcs import compute_upper_confidence_bound
from metatab.metalearning.sampler import OptunaRandomSampler
from metatab.metalearning.metadata_generator import MetadataGenerator
from metatab.metalearning.metadata_evaluator import MetadataEvaluator
from metatab.metalearning.utils import get_estimator_n_candidate_points
from metatab.hp_search.cv import CrossValidator
from metatab.hp_search.config import ConfigSearchCV

if TYPE_CHECKING:
    from metatab.preprocessing.types import ResolvedPreprocessingStrategy
    from metatab.utils.types import TunableEstimatorType, XType, YType
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams




class SearchCV:
    '''
    Class that implements HPs optimization using an inner cross-validation.

    Some key features:
    - Implements different optimazation algorithms: random, tpe and meta. 
    - Allows a meta-learning informed search via surrogate models using the "meta" algo.
    - Allows early stop on a validation set at fit time. This functionality needs the 
    classifier to implement an "eval_set-like" interface.
    - Allows to optionally refit the classifier with the best hyperparameters.
    - The search is not parallelizable even when the "random" algo is used. 
    

    Parameters:
        pipe (Pipeline):
            Pipeline object headed by a classifier which hps have to be optimized.
        
        type_estimator (TunableEstimatorType):
            String estimator type of the classifier head. 
            Info needed in meta-optimization (`meta` algo).
            
        preprocessing (ResolvedPreprocessingStrategy):
            Type of preprocessing used for the pipe object.
            Info needed in meta-optimization (`meta` algo).
        
        algo (Literal["random", "tpe", "meta"]): 
            Searching algorithm to use.
        
        sampler_function (Callable[[optuna.Trial], dict]):
            Optuna sampler function that carries the search space.

        n_iter (int): 
            Number of search iterations.
        
        n_cv_folds (int): 
            Number of cv folds.

        n_cv_repeats (int): 
            Number of cv repeats.
        
        seed (int): 
            Seed for reproducibility.
            Used in the standard optimization routes to draw points (random, tpe), 
            in the determination of the validation set when early stop is enabled,
            and in the inner-cross validation procedure (indipendently of `algo`).

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
            A dict unpackaged in the classifier fit call.
            If None (default) an empty dict is created.
            The dict keys must be in the pipeline format.
        
        meta_surrogate_model (None | str | Path, optional):
            Surrogate model to use in the meta-optimization scenario.
            If str or Path, then the object pointed by the path is used as surrogate model.
            This must be a joblib serialized object.
            If None the "default" surrogate model according to `type_estimator` is used instead.
            Ignored when `algo` is not "meta".

        meta_strategy (MetaStrategy, optional):
            Set the strategy used by the metalearning framework to select points.
            It has no effect when `algo` is not "meta".
            In detail the following `MetadataEvaluator` utilities are used:
            - "best": `propose_n_best`
            - "random_from_best": `propose_random_from_top`
            - "uniform_from_best": `propose_uniform_from_top`
            - "random_uniform_from_best": `propose_random_uniform_from_top`
            See the specific method for more details.

        meta_strategy_params (None | MetaStrategyParams, optional):
            Meta strategy specifics in form of dataclass.
            If None the default specifics are applied.

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

        params_as_object_columns_in_df_search (None | list[str], optional):
            The params that must be inserted in the df_search as object dtype columns.
            Ignored when `build_df_search` is False.

        refit_with_best_hps (None | bool, optional):
            Whether to refit the classifier with the best hps from the search.
            If None, the parameter is set via the global `ConfigSearchCV` configuration class.
    

    ## Attributes:

        best_params_ (dict):
            Best HPs configuration obtained from the tuning procedure.
        
        best_estimator_ (Pipeline):
            Refitted pipeline with the best hps configuration coming from the search.
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
        pipe: Pipeline,
        type_estimator: TunableEstimatorType,
        preprocessing: ResolvedPreprocessingStrategy,
        algo: Literal["random", "tpe", "meta"],
        sampler_function: Callable[[optuna.Trial], dict],
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
        meta_strategy: MetaStrategy = "best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_seed: int = 42,
        raise_error_during_search: None | bool = None,
        build_df_search: None | bool = None,
        params_as_object_columns_in_df_search: list[str] | None = None,
        refit_with_best_hps: None | bool = None
    ):
        self.pipe=pipe
        self.type_estimator=type_estimator
        self.preprocessing=preprocessing
        self.algo=algo
        self.sampler_function=sampler_function
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
        self.params_as_object_columns_in_df_search = params_as_object_columns_in_df_search

        self.cross_validator=CrossValidator(
            pipe=pipe,
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



    def fit(self, X: XType, y: YType) -> "SearchCV":
        '''Performs HPO. Returns the instance. '''
        self._X = X if isinstance(X, np.ndarray) else X.to_numpy()
        self._y = y if isinstance(y, np.ndarray) else y.to_numpy()
        
        self._dfs_cv_iter: list[pd.DataFrame] = []
        self.search_losses_: list[float] = []

        # with n_iter equal 1 we skip the point evaluation
        # since the single point is the best by definition.
        # Therefore we cannot build the df_search.
        if self.n_iter == 1:
            # we append nan since we do not evaluate the loss
            self.search_losses_.append(np.nan)
            self.build_df_search = False

        if self.algo == "meta":
            self._fit_with_meta_points()
        
        elif self.algo in ["random", "tpe"]:
            self._fit_with_standard_algo()

        else:
            raise ValueError("Unsupported optimization algorithm.")
        
        if self.build_df_search:
            self.df_search_ = self._build_df_search()

        # we refit on the original X and y to not influence sklearn expection
        # about the fit datatype which is checked at predict time 
        if self.refit_with_best_hps:
            best_estimator = deepcopy(self.pipe)
            set_params_into_clf(best_estimator, self.best_params_)   
            
            if self.early_stop_on_validation_set:
                self.best_estimator_, self.refit_time_ = fit_with_early_stop_on_validation_set(
                    pipe=best_estimator,
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
        n_candidate_points = max(get_estimator_n_candidate_points(self.type_estimator), self.n_iter) \
            if self.meta_strategy_params is None \
            else self.meta_strategy_params.n_candidate_points
        
        meta_generator = MetadataGenerator(
            sampler=OptunaRandomSampler(),
            mfe=CustomMFE(),
        )

        meta_generator.fit(
            X=self._X,
            y=self._y,
            sampler_function=self.sampler_function,
            seed=self.meta_seed 
        )

        metadata, candidate_points = meta_generator.generate(
            n_points=n_candidate_points,
            # add the preprocessing to the meta-data
            mfe_extract_kwargs = {"add_features": {"preprocessing": self.preprocessing}}
        )

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

        meta_evaluator = MetadataEvaluator(
            surrogate_framework=surrogate_model,
            acquisition_func=acquisition_func
        )

        _ = meta_evaluator.fit(metadata, candidate_points).evaluate_candidates()

        if self.meta_strategy == "best":
            points = meta_evaluator.propose_n_best(n_best=self.n_iter)
        
        elif self.meta_strategy == "random_from_best":
            # we use a ratio of 1 to 3 by default when possible, 
            # meaning we give "3 choices for point"
            top = min(self.n_iter * 3, n_candidate_points) \
                if self.meta_strategy_params is None \
                else self.meta_strategy_params.top
            
            # we use the instance seed to allow variability when not hardcoded in the supplied params
            points = meta_evaluator.propose_random_from_top(
                n_proposed=self.n_iter,
                top=top,
                seed=self.seed if self.meta_strategy_params is None else self.meta_strategy_params.seed
            )
        
        elif self.meta_strategy == "uniform_from_best":
            # we use a step of 3 by default when possible
            if self.meta_strategy_params is None:
                step_size = 3 if (n_candidate_points / self.n_iter) > 3 else 1
            else:
                step_size = self.meta_strategy_params.step_size
            
            points = meta_evaluator.propose_uniform_from_top(
                n_steps=self.n_iter,
                step_size=step_size
            )

        elif self.meta_strategy == "random_uniform_from_best":
            # we use a step of 3 by default when possible
            if self.meta_strategy_params is None:
                step_size = 3 if (n_candidate_points / self.n_iter) > 3 else 1
            else:
                step_size = self.meta_strategy_params.step_size

            # we use the instance seed to allow variability when not hardcoded in the supplied params
            points = meta_evaluator.propose_random_uniform_from_top(
                n_steps=self.n_iter,
                step_size=step_size,
                seed=self.seed if self.meta_strategy_params is None else self.meta_strategy_params.seed
            )

        # we do not evaluate the single point since is the best by definition
        if len(points) == 1:
            self.best_params_ = points[0]
            return None

        for point in points:
            _ = self._fit_point(point)

        losses = np.array(self.search_losses_)

        if np.isnan(losses).all():
            raise ValueError("All search iterations have failed.")
        
        self.best_params_ = points[np.nanargmin(losses)]


    
    def _fit_with_standard_algo(self) -> None:
        '''
        Optimize HPs with the random or tpe algo.
        Set the `best_params_` attribute.
        '''
        # ADDFEATURE: add logg capabilities to searchcv and time_limit.
        # disable optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.algo == "random":
            optuna_sampler = RandomSampler(seed=self.seed)
        else:
            optuna_sampler = TPESampler(
                n_startup_trials=20, # number of random init points, we double the default of 10
                seed=self.seed,
            )

        if self.n_iter == 1:
            # mock objective
            def objective(trial):
                self.sampler_function(trial)
                return 0
        else:
            def objective(trial):
                params = self.sampler_function(trial)
                return self._fit_point(params)
            
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                category=UserWarning, 
                message="Choices for a categorical distribution should be.*"
            )
            # we have only the logloss as metric so we minimize
            study = optuna.create_study(sampler=optuna_sampler, direction="minimize")
            study.optimize(func=objective, n_trials=self.n_iter, n_jobs=1)

        if not any([t.state == optuna.trial.TrialState.COMPLETE for t in study.trials]):
            raise ValueError("All iterations have failed.")
        
        # this resolve conditional logic returning a classifier-compatible point
        self.best_params_ = self.sampler_function(study.best_trial)



    def _fit_point(self, params: dict) -> float:
        '''
        Fit model using the input params and cv.
        Returns the cv loss.
        '''
        try:
            loss, df_cv_iter = self.cross_validator.fit(
                X=self._X, 
                y=self._y,
                params=params,
                agg="mean",
                build_df_cv=self.build_df_search,
                params_as_object_in_df_cv=self.params_as_object_columns_in_df_search
            )
            self._dfs_cv_iter.append(df_cv_iter)
            self.search_losses_.append(loss)
            return loss
            
        except Exception as e:
            # we re-raise the error if not tollerated or comes from the df_search bulding process
            if self.raise_error_during_search or isinstance(e, DFSearchBuildingError):
                raise ValueError(
                    f"The following error is encountered during the search: {e}"
                )
            # we enforce "search_losses_" to be of length n_iter
            self.search_losses_.append(np.nan)
            return np.nan
            


    def _build_df_search(self) -> pd.DataFrame:    
        # add search iter column and concat
        for i in range(len(self._dfs_cv_iter)):
            self._dfs_cv_iter[i]["search_iter"] = i
        return pd.concat(self._dfs_cv_iter, axis=0, ignore_index=True)