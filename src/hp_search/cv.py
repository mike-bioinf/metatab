from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from estimators.utils import fit_with_early_stop_on_validation_set
from estimators.params import HPS_MIXED_TYPES
from metatab_utils.general import add_broadcasted_objects_as_column
from hp_search.utils import set_params_into_clf

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from estimators.constants import Classifier
    
    


class CrossValidator:
    '''
    Handles the execution of the cross-validation procedures
    of classifier or pipeline objects.
    
    Parameters:
        clf_or_pipe (Classifier | Pipeline):
            Classifier or Pipeline object with a classifier as head, 
            which hps have to be optimized.

        clf_random_state_parameter (str):
            Name of the estimator random state parameter.

        early_stop_on_validation_set (bool):
            Whether to early stop on validation set(s).

        eval_set_parameter (str):
            Name of the eval_set parameter, 
            i.e. the parameter taking the validation set(s) at fit level.
            Ignored when "early_stop_on_validation_set" is False.
        
        validation_set_size (float):
            The ratio of the early stop validation set.
            Ignored when "early_stop_on_validation_set" is False.

        fit_classifier_kwargs (dict):
            A dict unpackaged in the classifier fit calls.
            The dict keys must be already adapted to the pipeline if any.

        metric (Literal["logloss"]):
            The performance metric to compute.

        n_splits (int):
            Number of cv splits.

        n_repeats (int):
            Number of cv repeats.
        
        seed (int):
            Seed for reproducibility.
    '''
    def __init__(
        self,
        clf_or_pipe: Classifier | Pipeline,
        clf_random_state_parameter: str,
        early_stop_on_validation_set: bool,
        eval_set_parameter: str,
        validation_set_size: float,
        fit_classifier_kwargs: dict,
        metric: Literal["logloss"],
        n_splits: int, 
        n_repeats: int, 
        seed: int
    ):
        self.clf_or_pipe=clf_or_pipe
        self.clf_random_state_parameter=clf_random_state_parameter
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs
        self.metric=metric
        self.n_splits=n_splits
        self.n_repeats=n_repeats
        self.seed=seed


    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict, 
        agg: Literal["mean", "sum"],
        collect_info: bool,
    ) -> tuple[float, pd.DataFrame|None]:
        '''
        Fit the cv procedure on the instance data.

        Parameters:
            X (pd.DataFrame): X data.
            y (pd.Series): y data.
            params (dict): Dict of classifier parameters. They must not follow the "pipeline format".
            agg (Literal["mean", "sum"]): How to aggregate the cv round performances.
            collect_info (bool): Whether to collect and return the cv info as dataframe.

        Returns:
            tuple[float,pd.DataFrame|None]:
            A tuple of the aggregated cv performances and the collected cv info. 
            The second term is None if `collect_info` is False.
        '''
        skf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, 
            n_repeats=self.n_repeats, 
            random_state=self.seed
        )

        cv_losses = []
        cv_results = []
        rng = np.random.default_rng(self.seed)        
        
        for iter_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            repeat = iter_idx // self.n_splits
            fold = iter_idx - (self.n_splits * repeat)

            # we create a copy of the clf/pipe at each cv round
            # to avoid specific classifier implementation problems
            # related to fitting multiple times the same instance.
            # (for example for catboost is not possible to set the parameters on a fitted instance)
            clf_or_pipe = deepcopy(self.clf_or_pipe)
            set_params_into_clf(clf_or_pipe, params)

            # we overwrite the classifier seed in order to maximize model entropy inside cv, 
            # while assuring uniformity between different cv runs.
            round_cv_seed = {self.clf_random_state_parameter: int(rng.integers(0, 2**32))}
            set_params_into_clf(clf_or_pipe, round_cv_seed, set_tabpfn_inference_config=False)
            
            X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

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
            if collect_info: cv_results.append({"repeat": repeat, "fold": fold, "loss": loss})

        array_cv_losses = np.array(cv_losses)
        agg_loss = np.mean(array_cv_losses) if agg == "mean" else np.sum(array_cv_losses)
        out = [agg_loss]
        
        if collect_info:
            df_info = pd.DataFrame(cv_results)
            # this is to block coercion when these dfs are concatenated with pd.concat.
            df_info = add_broadcasted_objects_as_column(
                df=df_info, 
                dictionary=params,
                convert_bool_to_str=False,
                convert_none_to_str=False,
                force_object_datatype=HPS_MIXED_TYPES,
                check_matching_keys_cols=True,
                check_non_builtin_types=True,
                copy=False
            )
            out.append(df_info)
        else:
            out.append(None)
        
        return tuple(out)


    def _compute_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.metric == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}.")