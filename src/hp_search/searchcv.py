import time
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Literal, Any
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, rand, fmin, space_eval
from hyperopt.pyll.stochastic import sample
from estimators.constants import Classifier
from estimators.utils import fit_with_early_stop_on_validation_set
from hp_search.utils import ConfigSearchCV, set_params_into_clf
from hp_search.cv import CrossValidator
from metalearning.metafeatures import extract_metafeatures
from metalearning.database.utils import query_surrogate_framework
from metalearning.acquisition_funcs import compute_upper_confidence_bound




class SearchCV:
    '''
    Class that implements HPs optimization via random search or
    tpe methods with (repeated) cross-validation.

    Allows a meta-learning informed search via surrogate models using "meta" algo.

    Allows early stop on validation set at fit time, only if the classifier
    implements this feature in its API via the "eval_set interface".

    It optionally refit the classifier/pipeline with the best hyperparameters.
    Exposes the "predict_proba" method of the refitted object.

    The search is not parallelized even when the "random" algo is selected. 
    
    -------------------------------
    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Classifier or Pipeline object with a classifier as head, 
            which hps have to be optimized.
        
        type_clf_or_pipe_preprocessing (Literal["base", "density_filter", "pca"] | None):
            Type of preprocessing used for the clf_or_pipe object.
            Needed only by the `meta` algo to propose candidate points.
        
        algo (Literal["random", "tpe", "meta"]):
            Type of searching algorithm to use.
        
        params_distributions (dict):
            Search space.

        n_iter (int):
            Number of search iterations.
        
        n_cv_splits (int):
            Number of cv splits.

        n_cv_repeats (int):
            Number of cv repeats.
        
        seed (int):
            Seed for reproducibility.
        
        random_state_parameter (str):
            Name of the estimator random state parameter.
        
        metric_to_minimize (Literal["logloss"]):
            The metric to minimize in the search.

        early_stop_on_validation_set (bool):
            Whether to early stop on validation set(s).

        eval_set_parameter (str, optional):
            Name of the eval_set parameter, 
            i.e. the parameter taking the validation set(s) at fit level.
            Ignored when "early_stop_on_validation_set" is False.
        
        validation_set_size (float, optional):
            The ratio of the early stop validation set.
            Inside cv this set is taken from the training portion.
            Ignored when "early_stop_on_validation_set" is False.

        fit_classifier_kwargs (None | dict, optional):
            A dict unpackaged in the classifier fit calls.
            If None (default) an empty dict is created.
            The dict keys must be already adapted to the pipeline if any.
        
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
       

    Attributes:
    ------------------------------------
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
        type_clf_or_pipe_preprocessing: Literal["base", "density_filter", "pca"] | None,
        algo: Literal["random", "tpe", "meta"],
        params_distributions: dict,
        n_iter: int,
        n_cv_repeats: int,
        n_cv_splits: int,
        random_state_parameter: str,
        seed: int,
        metric_to_minimize: Literal["logloss"],
        early_stop_on_validation_set: bool,
        eval_set_parameter: str = "eval_set",
        validation_set_size: float = 0.3,
        fit_classifier_kwargs: None | dict = None,
        raise_error_during_search: None | bool = None,
        build_df_search: None | bool = None,
        refit_with_best_hps: None | bool = None,
        save_realtime_df_search_filepath: None | str | Path = None
    ):
        self.clf_or_pipe=clf_or_pipe
        self.type_clf_or_pipe_preprocessing=type_clf_or_pipe_preprocessing
        self.algo=algo
        self.params_distributions=params_distributions
        self.random_state_parameter=random_state_parameter
        self.n_iter=n_iter
        self.n_cv_repeats=n_cv_repeats
        self.n_cv_splits=n_cv_splits
        self.seed=seed
        self.metric_to_minimize=metric_to_minimize
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs if fit_classifier_kwargs else {}
        
        self.cross_validator=CrossValidator(
            clf_or_pipe=clf_or_pipe,
            clf_random_state_parameter=random_state_parameter,
            early_stop_on_validation_set=early_stop_on_validation_set,
            eval_set_parameter=eval_set_parameter,
            validation_set_size=validation_set_size,
            fit_classifier_kwargs=self.fit_classifier_kwargs, # here we must always pass a dict
            metric=metric_to_minimize,
            n_splits=n_cv_splits,
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
        '''
        Performs HPO and optinally refit the estimator with the best hps.      
        Returns the instance.
        '''
        self._X = X
        self._y = y
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
            self._metafeatures = extract_metafeatures(X, y)
            self._surrogate_framework = query_surrogate_framework(self.clf_or_pipe)
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
        points = self._propose_meta_points(
            n_candidate_points=5000,
            # with "meta" algo n_iter set the number of evaluated points
            n_points_to_propose=self.n_iter,
            acquisition_function="UCB"
        )

        # we do not evaluate the single point since is the best by definition
        if len(points) == 1:
            self.best_params_ = points[0]
            return None

        for point in points:
            _ = self._fit_point(
                point,
                apply_hyperopt_corrections=False,  # the proposed points are already corrected
                returns_type="simple"
            )

        losses = np.array(self.search_losses_)

        if np.isnan(losses).all():
            raise ValueError("All search iterations have failed.")
        
        self.best_params_ = points[np.nanargmin(losses)]
    


    def _fit_with_standard_algo(self) -> None:
        '''
        Optimize HPs with the random or tpe algo.
        Set the `best_params_` attribute.
        '''
        # with n_iter to 1 the sampling is always random 
        # and the drawn point is the best by definition
        if self.n_iter == 1:
            self.best_params_ = self._apply_hyperopt_corrections_to_sampled_point(
                sample(self.params_distributions, np.random.default_rng(self.seed))
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

        fit_point_fn = partial(
            self._fit_point,
            apply_hyperopt_corrections=True,
            returns_type="hyperopt"
        )

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
        self.best_params_ = self._apply_hyperopt_corrections_to_sampled_point(
            space_eval(self.params_distributions, best)
        )



    def _fit_point(
        self, 
        params: dict,
        apply_hyperopt_corrections: bool,
        returns_type: Literal["hyperopt", "simple"]
    ) -> dict | float:
        '''
        Fit using the input tune space point.

        Parameters:
        params (dict): dict of hps to use (tune space point).
        apply_hyperopt_corrections (bool):
            Whether to apply the hyperopt corrections to the point.
        returns_type (Literal["hyperopt", "simple"]):
            Whether returns a hyperopt compatible result or a simpler one.
            In the first case the function returns a dict with hyperopt
            compatible info, in the second only the loss.
        '''
        try:
            if apply_hyperopt_corrections:
                params = self._apply_hyperopt_corrections_to_sampled_point(params)

            loss, df_cv_info = self.cross_validator.fit(
                X=self._X, 
                y=self._y,
                params=params,
                agg="mean",
                collect_info=self.build_df_search
            )

            # the code should not fail a single time from here, 
            # but if it happens then we have external bug/problems
            # masked as failed optimization iteration.
            # (for example no space and then yes on disk)
            if self.build_df_search:
                self._dfs_info_iter.append(df_cv_info)
                # here a bit inefficient since we rebuild multiple times
                if self.save_realtime_df_search_filepath:
                    self.df_search_ = self._build_df_search()
                    self.df_search_.to_csv(self.save_realtime_df_search_filepath, sep="\t", index=False)

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



    def _propose_meta_points(
        self,
        n_candidate_points: int,
        n_points_to_propose: int,
        acquisition_function: Literal["UCB"]
    ) -> list[dict[str, Any]]:
        '''
        Propose the most promising points on the tune space
        based on a surrogate model and an acquisition function.

        Parameters:
            n_candidate_points (int): 
                Number of points to draw as candidates.
            n_points_to_propose (int): 
                Number of points returned by the utility.
            acquisition_function (Literal["UCB"]): 
                Select the function evaluating the 
                promissingness of the candidate points.

        Returns:
            list[dict[str,Any]]: 
            A list of dict where each dict is a point in the tune space.
        '''     
        rng_candidates = np.random.default_rng(self.seed)
        
        candidate_points = [
            self._apply_hyperopt_corrections_to_sampled_point(
                sample(self.params_distributions, rng_candidates)
            )
            for _ in range(n_candidate_points)
        ]
     
        df_candidate_points = pd.DataFrame(candidate_points)
        df_candidate_points["preprocessing"] = self.type_clf_or_pipe_preprocessing

        for metafeature, value in self._metafeatures.items():
            df_candidate_points[metafeature] = value

        pred_values, pred_uncertainty = self._surrogate_framework.predict(df_candidate_points)
        
        if acquisition_function == "UCB":
            promisingness = compute_upper_confidence_bound(
                pred_values, 
                pred_uncertainty,
                k="infer", 
                mean_direction="lower_is_better", # we currently use only the logloss
                n_points=n_points_to_propose
            )
        else:
            raise ValueError(f"'acquisition_function' must be equal to 'UCB'.")

        # argsort works in the increasing order (last index --> index of the greatest value)
        top_idx = np.argsort(promisingness, stable=True)[-n_points_to_propose:]
        selected_points = [candidate_points[idx] for idx in top_idx]
        return selected_points
        


    @staticmethod
    def _apply_hyperopt_corrections_to_sampled_point(params: dict[str, Any]) -> dict[str, Any]:
        '''
        Apply general hyperopt level correction to the sampled params.
        These corrections come from specific quirks of hyperopt.
        The corrections are done in place.

        In particular the following aspects are addressed:
        - automatic conversion of sampled list to tuple. 
            To distinguish between original and converted tuple we cast 
            the specific parameters explicitly.
        '''
        tuple_to_list_parameters = [
            "inference_config__PREPROCESS_TRANSFORMS"
        ]
        
        for param_to_convert in tuple_to_list_parameters:
            if param_to_convert in params.keys():
                params[param_to_convert] = list(params[param_to_convert])

        return params
    


    def _build_df_search(self) -> pd.DataFrame:    
        # add search iter column and concat
        for i in range(len(self._dfs_info_iter)):
            self._dfs_info_iter[i]["search_iter"] = i
        return pd.concat(self._dfs_info_iter, axis=0, ignore_index=True)

    

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)