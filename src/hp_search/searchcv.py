import warnings
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from typing import Literal, Any
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, rand, fmin, space_eval
from hyperopt.pyll.stochastic import sample
from metatab_utils.general import add_broadcasted_objects_as_column
from estimators.constants import Classifier
from estimators.utils import fit_with_early_stop_on_validation_set
from estimators.params import HPS_MIXED_TYPES
from hp_search.utils import ConfigSearchCV
from _paper.hp_metalearning.metafeatures import extract_metafeatures
from _paper.hp_metalearning.database.utils import query_surrogate_framework
from _paper.hp_metalearning.acquisition_funcs import compute_upper_confidence_bound




class SearchCV:
    '''
    Class that implements HPs optimization via random search or
    tpe methods with (repeated) cross-validation.

    Allows a meta-learning informed search via surrogate models using "meta" algo.

    Allows early stop on validation set at fit time, only if the classifier
    implements this feature in its API via the "eval_set interface".

    It always refit the classifier/pipeline with the best hyperparameters.
    Exposes the "predict_proba" method of the refitted object.

    The search is not parallelized even when the "random" algo is selected. 
    
    -------------------------------
    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Classifier or Pipeline object with a classifier as head, 
            which hps have to be optimized.
        
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
        
        validation_set_size (flot, optional):
            The ratio of the early stop validation set.
            Inside cv this set is taken from the training portion.
            Ignored when "early_stop_on_validation_set" is False.

        fit_classifier_kwargs (None | dict, optional):
            A dict unpackaged in the classifier fit calls.
            If None (default) an empty dict is created.
            The dict keys must be already adapted to the pipeline if any.
        
        build_df_search (None | bool, optional):
            Whether to build the DataFrame with complete search information.  
            If False, the required information is not stored.  
            This step can be indeed memory and time consuming.
            If None, the parameter is set via a global configuration class.

            
    Attributes:
    ------------------------------------
        best_params_ (dict):
            Best HPs configuration obtained from the tuning procedure.
        
        best_estimator_ (Classifier | Pipeline):
            Refitted classifier/pipeline with the best hps configuration
            coming from the search.

        df_search_ (pd.DataFrame):
            Dataframe with the search info (hps and loss) at cv-fold level.
            Does not contain info about the failed iterations.
            Keep in mind that the the completed iterations are numerically 
            sequentially labeled at the end of the search ("search_iter" column).
            This means that if point n2 in the search fails, then point n3 is reported as 2 in the df.
            The attribute is set only when "build_df_search" flag is True.
        
        search_losses_ (list):
            List of the losses registered during the search.
            Contains np.nan for failed iterations.
            The search order is respected.

        refit_time_ (float):
            Time of refit on the best configuration in seconds.
    '''
    def __init__(
        self,
        *,
        clf_or_pipe: Classifier | Pipeline,
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
        build_df_search: None | bool = None
    ):
        self.clf_or_pipe=clf_or_pipe
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
        self.build_df_search=ConfigSearchCV.build_df_search if build_df_search is None else build_df_search



    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SearchCV":
        '''
        Performs HPO and refit the estimator with the best hps.
        Set the "best_params_" and "best_estimator_" attributes.        
        Returns the instance.
        '''
        self._X = X
        self._y = y
        
        self.search_losses_: list[float] = []
        # each dict of "_iter_params_cv" is associated to a list of "_iter_cv_results"
        self._iter_cv_results: list[list[dict]] = []
        self._iter_params_cv: list[dict] = []
        self._number_completed_iter = 0

        if self.algo == "meta":
            self._metafeatures = extract_metafeatures(X, y)
            self._surrogate_framework = query_surrogate_framework(self.clf_or_pipe)
            self._fit_with_meta_points()
        elif self.algo in ["random", "tpe"]:
            self._fit_with_standard_algo()
        else:
            raise ValueError("Unsupported optimization algorithm.")
        
        if self.build_df_search:
            self.df_search_ = self._build_df_search()

        # refit with the best point
        best_estimator = deepcopy(self.clf_or_pipe)
        self._set_params_into_clf(best_estimator, self.best_params_)   
        
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
        Set the best_params_ attribute.
        '''
        points = self._propose_meta_points(
            n_candidate_points=5000,
            # with "meta" algo n_iter set the number of evaluated points
            n_points_to_propose=self.n_iter,
            acquisition_function="UCB"
        )

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
        Set the best_params_ attribute.
        '''
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
            loss = self._cross_val_score(params)
            self.search_losses_.append(loss)
            if returns_type == "hyperopt":
                return {"loss": loss, "status": STATUS_OK}
            else:
                return loss
        except Exception as e:
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
            list[dict[str, Any]]: 
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
    


    def _cross_val_score(
        self,
        params: dict, 
        agg: Literal["mean", "sum"] = "mean"
    ) -> float:
        '''
        Custom implementation of cross_val_score allowing 3-sets ripartitions.
        Returns the total cv loss as sum or mean of internal round losses.
        '''
        skf = RepeatedStratifiedKFold(
            n_splits=self.n_cv_splits, 
            n_repeats=self.n_cv_repeats, 
            random_state=self.seed
        )

        cv_losses = []
        cv_results = []
        rng_cv = np.random.default_rng(self.seed)        
        
        for iter_idx, (train_idx, test_idx) in enumerate(skf.split(self._X, self._y)):
            repeat = iter_idx // self.n_cv_splits
            fold = iter_idx - (self.n_cv_splits * repeat)

            # we create a copy of the clf/pipe at each cv round
            # to avoid specific classifier implementation problems
            # related to fitting multiple times the same instance.
            # (for example for catboost is not possible to set the parameters on a fitted instance)
            clf_or_pipe = deepcopy(self.clf_or_pipe)
            self._set_params_into_clf(clf_or_pipe, params)

            # we overwrite the classifier seed 
            # in order to maximize model entropy inside cv, 
            # while assuring uniformity between different cv runs.
            round_cv_seed = {self.random_state_parameter: int(rng_cv.integers(0, 2**32))}
            self._set_params_into_clf(clf_or_pipe, round_cv_seed, set_tabpfn_inference_config=False)
            
            X_train, y_train = self._X.iloc[train_idx, :], self._y.iloc[train_idx]
            X_test, y_test = self._X.iloc[test_idx, :], self._y.iloc[test_idx]

            if self.early_stop_on_validation_set:
                clf_or_pipe = fit_with_early_stop_on_validation_set(
                    clf_or_pipe=clf_or_pipe,
                    X=X_train,
                    y=y_train,
                    seed=self.seed,
                    validation_set_size=self.validation_set_size,
                    eval_set_parameter=self.eval_set_parameter,
                    fit_classifier_kwargs=self.fit_classifier_kwargs
                )
            else:
                clf_or_pipe.fit(X_train, y_train, **self.fit_classifier_kwargs)

            pred_proba = clf_or_pipe.predict_proba(X_test)
            loss = self._compute_loss_score(pred_proba, y_test)
            cv_losses.append(loss)

            if self.build_df_search:
                cv_results.append({"repeat": repeat, "fold": fold, "loss": loss})

        if self.build_df_search:
            # adding the results at the end allows to avoid adding the results of failing cv
            self._iter_cv_results.append(cv_results)
            self._iter_params_cv.append(params)
            self._number_completed_iter += 1
        
        return np.mean(cv_losses) if agg == "mean" else np.sum(cv_losses)



    def _build_df_search(self) -> pd.DataFrame:
        dfs_iters = []
        
        for i in range(self._number_completed_iter):
            df_iter = pd.DataFrame(self._iter_cv_results[i])
            dict_iter_params = self._iter_params_cv[i]
            dict_iter_params["search_iter"] = i
            
            # this is because concatenation between same named column with 
            # different dtypes causes coection when possible.
            df_iter = add_broadcasted_objects_as_column(
                df=df_iter, 
                dictionary=dict_iter_params,
                convert_bool_to_str=False,
                convert_none_to_str=False,
                force_object_datatype=HPS_MIXED_TYPES,
                check_matching_keys_cols=True,
                check_non_builtin_types=True,
                copy=False
            )

            dfs_iters.append(df_iter)
        
        df_search_ = pd.concat(dfs_iters, axis=0, ignore_index=True)
        return df_search_



    def _set_params_into_clf(
        self, 
        clf_or_pipe: Classifier | Pipeline, 
        params: dict[str, Any],
        set_tabpfn_inference_config: bool = True
    ) -> None:
        '''
        Set the parameters into the classifier.
        The method works with all type of classifiers even when they head pipeline objects.
        The method overwrites the pre-existent parameters values for the ones specified in params.
        For tabpfn classifiers is possible to micro manage 
        the setting of the "inference_config__" marked parameters.
        '''
        clf = clf_or_pipe[-1] if isinstance(clf_or_pipe, Pipeline) else clf_or_pipe
        
        if isinstance(clf, TabPFNClassifier):
            if "inference_config" in params.keys():
                raise KeyError(
                    "The inference_config parameter cannot be handled explicity.",
                    "Instead its keys must be passed as normal parameters marked with the 'inference_config__' prefix."
                )

            inference_config = {}
            cleaned_params = {}
            
            for k, v in params.items():
                if k.startswith("inference_config__"):
                    inference_config[f"{k.removeprefix("inference_config__")}"] = v
                else:
                    cleaned_params[k] = v

            if set_tabpfn_inference_config:
                if not inference_config:
                    warnings.warn(
                        message=(
                            "Derived an empty inference_config dict. "
                            "It will overwrite the classifier's existing inference_config."
                        ),
                        category=UserWarning
                    )
                clf.set_params(inference_config=inference_config, **cleaned_params)
            else:
                if inference_config:
                    warnings.warn(
                         message=(
                            "Derived a non-empty inference_config dict, but since "
                            "set_tabpfn_inference_config=False, it will be ignored."
                        ),
                        category=UserWarning
                    )
                clf.set_params(**cleaned_params)
        
        else:
            clf.set_params(**params)

    

    def _compute_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.metric_to_minimize == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_to_minimize}.")
        
    

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)