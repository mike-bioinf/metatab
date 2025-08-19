import traceback
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from typing import Literal, Callable
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_is_fitted
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, rand, fmin, space_eval
from estimators.types import Classifier

from estimators.utils import (
    fit_with_early_stop_on_validation_set,
    add_string_to_params
)



class SearchCV:

    def __init__(
        self,
        *,
        clf_or_pipe: Classifier | Pipeline,
        algo: Literal["random", "tpe"], 
        params_distributions: dict,
        n_iter: int,
        n_cv_repeats: int,
        n_cv_splits: int,
        random_state_parameter: str,
        seed: int,
        metric_to_minimize: Literal["logloss"],
        early_stop_on_validation_set: bool,
        eval_set_parameter: str = "eval_set",
        validation_set_size: float = 0.3
    ):
        '''
        Class that implements HPs optimization via random search or
        tpe methods with (repeated) cross-validation.

        Allows early stop on validation set at fit time, only if the classifier
        implements this feature in its API via the "eval_set interface".

        It always refit the classifier/pipeline with the best hyperparameters.
        Exposes the "predict_proba" method of the refitted object.

        The search is not parallelizable even when the "random" algo is selected. 
        
        Parameters:
        -------------------------------
            clf_or_pipe (Classifier | Pipeline):
                Classifier or Pipeline object with a classifier as head, 
                which hps have to be optimized.
            
            algo (Literal["random", "tpe"]):
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
                Whether to early stop on validation sets.

            eval_set_parameter (str, optional):
                Name of the eval_set parameter, 
                i.e. the parameter taking the validation sets at fit level.
                Ignored when "early_stop_on_validation_set" is False.
            
            validation_set_size (flot, optional):
                The ratio of the early stop validation set.
                Inside cv this set is taken from the training portion.
                Ignored when "early_stop_on_validation_set" is False.

        Attributes:
        ------------------------------------
            best_params_ (dict):
                Best HPs configuration obtained from the tuning procedure.

            best_estimator_ (Classifier | Pipeline):
                Refitted classifier/pipeline with the best hps configuration
                coming from the search.

            trials_ (Trials):
                Trials object with search info.
        '''
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
        self.add_string_to_params_func=self._get_add_string_to_params_func(clf_or_pipe)


    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SearchCV":
        '''
        Performs HPO and always refit the estimator with the best hps.
        Returns the instance.
        '''
        self.X = X
        self.y = y
        algo_fn = tpe.suggest if self.algo == "tpe" else rand.suggest
        trials = Trials()
        
        best = fmin(
            fn=self,
            space=self.params_distributions,
            algo=algo_fn,
            max_evals=self.n_iter,
            trials=trials,
            rstate=np.random.default_rng(self.seed),
            verbose=False        
        )
        
        self.trials_ = trials
        self.best_params_ = space_eval(self.params_distributions, best)
        best_estimator = deepcopy(self.clf_or_pipe)
        self._set_params_into_clf(best_estimator, self.best_params_)

        # refit 
        if self.early_stop_on_validation_set:
            self.best_estimator_ = fit_with_early_stop_on_validation_set(
                clf_or_pipe=best_estimator,
                X=X,
                y=y,
                seed=self.seed,
                validation_set_size=self.validation_set_size,
                eval_set_parameter=self.eval_set_parameter
            )
        else:
            self.best_estimator_ = best_estimator.fit(X, y)
        
        return self


    def __call__(self, params: dict):
        '''Take in input the sampled point from the hp space'''
        try:
            loss = self._cross_val_score(params)
            return {"loss": loss, "status": STATUS_OK}
        except Exception as e:
            return {
                "loss": np.nan, 
                "status": STATUS_FAIL,
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        

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

        rng_cv = np.random.default_rng(self.seed)        
        loss_scores = []
        
        for train_idx, test_idx in skf.split(self.X, self.y):
            # we create a copy of the clf/pipe at each cv round
            # to avoid specific classifier implementation problems
            # related to fitting multiple times the same instance.
            # (for example for catboost is not possible to set the parameters on a fitted instance)
            clf_or_pipe = deepcopy(self.clf_or_pipe)
            self._set_params_into_clf(clf_or_pipe, params)
            # we overwrite the classifier seed 
            # in order to maximize model entropy inside cv, 
            # while assuring uniformity between different cv runs.
            round_cv_seed = {self.random_state_parameter: rng_cv.integers(0, 2**32)}
            self._set_params_into_clf(clf_or_pipe, round_cv_seed)
            
            X_train, y_train = self.X.iloc[train_idx, :], self.y.iloc[train_idx]
            X_test, y_test = self.X.iloc[test_idx, :], self.y.iloc[test_idx]

            if self.early_stop_on_validation_set:
                clf_or_pipe = fit_with_early_stop_on_validation_set(
                    clf_or_pipe=clf_or_pipe,
                    X=X_train,
                    y=y_train,
                    seed=self.seed,
                    validation_set_size=self.validation_set_size,
                    eval_set_parameter=self.eval_set_parameter
                )
            else:
                clf_or_pipe.fit(X_train, y_train)

            pred_proba = clf_or_pipe.predict_proba(X_test)
            loss_scores.append(self._compute_loss_score(pred_proba, y_test))

        sum_losses = sum(loss_scores)
        
        if agg == "sum":
            return sum_losses
        else:
            return sum_losses/len(loss_scores) 


    def _get_add_string_to_params_func(self, clf_or_pipe: Classifier | Pipeline) -> Callable:
        '''
        Derives and returns the function that manages 
        the addiction of the classifier name to the dict of parameters.
        '''
        if isinstance(clf_or_pipe, Pipeline):
            name_classifier = f"{clf_or_pipe.steps[-1][0]}__"
            return partial(add_string_to_params, string=name_classifier)
        else:
            return self._identity_params


    @staticmethod
    def _identity_params(params: dict):
        return params


    def _set_params_into_clf(self, clf_or_pipe: Classifier | Pipeline, params: dict) -> None:
        '''Set the parameters into the classifier whether it is in a pipeline or not'''
        clf_or_pipe.set_params(**self.add_string_to_params_func(params))

        
    def _compute_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.metric_to_minimize == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_to_minimize}.")
        
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)
