from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from metatab.utils.core import fit_with_early_stop_on_validation_set, set_params_into_clf
from metatab.utils.general import add_broadcasted_objects_as_column, ensure_or_create
from metatab.utils.exceptions import DFSearchBuildingError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline



class CrossValidator:
    '''
    Handles the execution of the cross-validation procedure.
    
    Parameters:
        pipe (Pipeline):
            Pipeline object with a classifier as head,
            which hps have to be optimized.

        clf_random_state_parameter (str):
            Name of the classifier random state parameter.

        early_stop_on_validation_set (bool):
            Whether to early stop on validation set(s).

        eval_set_parameter (str | None):
            Name of the eval_set parameter, i.e. the parameter 
            taking the validation set(s) at fit level. 
            If None then `early_stop_on_validation_set` must be False.
            
        validation_set_size (float):
            The ratio of the early stop validation set.
            Ignored when "early_stop_on_validation_set" is False.

        fit_classifier_kwargs (dict):
            A dict unpackaged in the classifier fit calls.
            The dict keys must be in the pipeline format.

        metric (Literal["logloss"]):
            The performance metric to compute.

        n_folds (int):
            Number of cv splits.

        n_repeats (int):
            Number of cv repeats.
        
        seed (int):
            Seed for reproducibility.
    '''
    def __init__(
        self,
        pipe: Pipeline,
        clf_random_state_parameter: str,
        early_stop_on_validation_set: bool,
        eval_set_parameter: str | None,
        validation_set_size: float,
        fit_classifier_kwargs: dict,
        metric: Literal["logloss"],
        n_folds: int, 
        n_repeats: int, 
        seed: int
    ):
        self.pipe=pipe
        self.clf_random_state_parameter=clf_random_state_parameter
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs
        self.metric=metric
        self.n_folds=n_folds
        self.n_repeats=n_repeats
        self.seed=seed


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict, 
        agg: Literal["mean", "sum"],
        build_df_cv: bool,
        params_as_object_in_df_cv: list[str] | None
    ) -> tuple[float, pd.DataFrame|None]:
        '''
        Fit the cv procedure on the instance data.

        Parameters:
            X (np.ndarray): X data.
            y (np.ndarray): y data.
            params (dict): Dict of classifier parameters. They must NOT follow the "pipeline format".
            agg (Literal["mean", "sum"]): How to aggregate the cv round performances.
            build_df_cv (bool): Whether to collect and return the cv info as dataframe.
            params_as_object_in_df_cv (list[str] | None): 
                Parameters to be inserted as object dtyped columns in the df_cv. 
                Ignored when 'build_df_cv' is False.

        Returns:
            tuple[float,pd.DataFrame|None]:
            A tuple of the aggregated cv performances and the collected cv info. 
            The second term is None if `build_df_cv` is False.
        '''
        skf = RepeatedStratifiedKFold(
            n_splits=self.n_folds, 
            n_repeats=self.n_repeats, 
            random_state=self.seed
        )

        cv_losses = []
        cv_results = []
        rng = np.random.default_rng(self.seed)        
        
        for iter_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            repeat = iter_idx // self.n_folds
            fold = iter_idx - (self.n_folds * repeat)

            # we create a copy of the pipe at each cv round
            # to avoid specific classifier implementation problems
            # related to fitting multiple times the same instance.
            # (for example for catboost is not possible to set the parameters on a fitted instance)
            pipe = deepcopy(self.pipe)
            # we overwrite the classifier seed in order to maximize model entropy inside cv, 
            # while assuring uniformity between different cv runs.
            params = {**params, self.clf_random_state_parameter: int(rng.integers(0, 2**32))}
            set_params_into_clf(pipe, params, set_tabpfn_inference_config=True, finalize_tabpfn_model_path=True)
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            if self.early_stop_on_validation_set:
                pipe = fit_with_early_stop_on_validation_set(
                    pipe=pipe,
                    X=X_train,
                    y=y_train,
                    seed=self.seed,
                    validation_set_size=self.validation_set_size,
                    eval_set_parameter=self.eval_set_parameter,
                    fit_classifier_kwargs=self.fit_classifier_kwargs
                )
            else:
                pipe.fit(X_train, y_train, **self.fit_classifier_kwargs)

            pred_proba = pipe.predict_proba(X_test)
            loss = self._compute_loss_score(pred_proba, y_test)
            cv_losses.append(loss)
            if build_df_cv: cv_results.append({"repeat": repeat, "fold": fold, "loss": loss})

        array_cv_losses = np.array(cv_losses)
        agg_loss = np.mean(array_cv_losses) if agg == "mean" else np.sum(array_cv_losses)
        
        if build_df_cv:
            try:
                df_cv = add_broadcasted_objects_as_column(
                    df=pd.DataFrame(cv_results),
                    dictionary=params,
                    convert_bool_to_str=False,
                    convert_none_to_str=False,
                    force_object_datatype=ensure_or_create(params_as_object_in_df_cv, list),
                    check_matching_keys_cols=True,
                    check_non_builtin_types=True,
                    copy=False
                )
                    
            except Exception as e:
                raise DFSearchBuildingError(
                    f"Encountered the following error during df_search building process: {e}"
                )

            return (agg_loss, df_cv)
        else:
            return (agg_loss, None)
        
    
    def _compute_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.metric == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}.")