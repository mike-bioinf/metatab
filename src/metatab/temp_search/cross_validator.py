from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Literal
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from metatab.classifiers.registry import ClassifierSpec
from metatab.utils.core import fit_with_early_stop_on_validation_set
from metatab.utils.general import add_broadcasted_objects_as_column, ensure_or_create
from metatab.utils.exceptions import PipelineFitError
from metatab.utils.pipeline import build_pipeline

if TYPE_CHECKING:
    from metatab.utils.types import XType, YType
    from metatab.preprocessing.types import PreprocessingStrategy



class CrossValidator:
    '''
    Score pipeline configurations using a cross-validation procedure.
    Raises "PipelineFitError" when the pipeline fit process fails.
    The instance stores the cv info in the "dfs_cv" attribute when "store_cv_info" is True.
    
    Parameters:
        X (XType): X data.
        
        y (YType): y data.

        n_folds (int): 
            Number of cv folds.

        n_repeats (int):
            Number of cv repeats.

        metric (Literal["logloss"]):
            The performance metric to compute.

        metric_aggregation (Literal["mean", "sum"]):
            Aggregation strategy used to compute whole cv value.
        
        seed (int):
            Seed for reproducibility.

        validation_set_size (float):
            The validation set ratio.
            Ignored for classifiers that do not use validation sets.
        
        store_cv_info (bool):
            Whether to store the cv info.
            If True the info are stored as lists of DataFrame objects in "dfs_cv" attribute.
    '''
    def __init__(
        self,
        X: XType,
        y: YType,
        n_folds: int, 
        n_repeats: int, 
        metric: Literal["logloss"],
        metric_aggregation: Literal["mean", "sum"],
        seed: int,
        validation_set_size: float,
        store_cv_info: bool
    ):
        self.X = X if isinstance(X, np.ndarray) else X.to_numpy()
        self.y = y if isinstance(y, np.ndarray) else y.to_numpy()
        self.n_folds=n_folds
        self.n_repeats=n_repeats
        self.metric=metric
        self.metric_aggregation=metric_aggregation
        self.validation_set_size=validation_set_size
        self.seed=seed
        self.store_cv_info=store_cv_info
        self.dfs_cv = []


    def score(
        self, 
        preprocessing: PreprocessingStrategy, 
        hps: dict, 
        classifier_spec: ClassifierSpec,
        add_fixed_hps: bool
    ) -> float:
        '''
        Score a pipeline configuration.
        Note that the configuration is passed through its attributes. 

        Parameters:
            preprocessing (PreprocessingStrategy): preprocessing.
            hps (dict): classifier hps.
            classifier_spec (ClassifierSpec): classifier dataclass.
            add_fixed_hps (bool): 
                Whether to retrieve the fixed hps from classifier_spec and 
                add them to the hps passed in input to this function to instanziate the classifier.
                Note that the fixed hps are in every case not stored in the instance.

        Returns:
           float: The aggregated cv loss.
        '''
        rng = np.random.default_rng(self.seed)        
        
        skf = RepeatedStratifiedKFold(
            n_splits=self.n_folds, 
            n_repeats=self.n_repeats, 
            random_state=self.seed
        )

        cv_losses = []
        cv_results = []
        cv_pred_proba = []

        for iter_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, y_train = self.X[train_idx, :], self.y[train_idx]
            X_test, y_test = self.X[test_idx, :], self.y[test_idx]
            repeat = iter_idx // self.n_folds
            fold = iter_idx - (self.n_folds * repeat)
            
            if add_fixed_hps:
                full_hps = {**hps, **classifier_spec.default_params}
            else:
                full_hps = hps

            pipe = build_pipeline(preprocessing, full_hps, classifier_spec, int(rng.integers(0, 2**32)))
            
            try:
                if classifier_spec.early_stop_on_validation_set:
                    pipe = fit_with_early_stop_on_validation_set(
                        pipe=pipe,
                        X=X_train,
                        y=y_train,
                        seed=self.seed,
                        validation_set_size=self.validation_set_size,
                        eval_set_parameter="eval_set" ## REFACTOR: we can remove this info???
                    )
                else:
                    pipe.fit(X_train, y_train)
            except Exception as e:
                raise PipelineFitError("Pipeline fitted process failed.") from e

            pred_proba = pipe.predict_proba(X_test)
            loss = self._compute_loss_score(pred_proba, y_test)
            cv_losses.append(loss)

            if self.store_cv_info:
                cv_pred_proba.append(pred_proba)
                cv_results.append({"repeat": repeat, "fold": fold, "loss": loss})

        array_cv_losses = np.array(cv_losses)
        agg_loss = np.mean(array_cv_losses) if self.metric_aggregation == "mean" else np.sum(array_cv_losses)
        
        if self.store_cv_info:
            df_cv = pd.DataFrame(cv_results)
            df_cv["pred_proba"] = cv_pred_proba
            df_cv["classifier"] = classifier_spec.type_classifier
            df_cv["preprocessing"] = preprocessing
            df_cv = add_broadcasted_objects_as_column(
                df=df_cv,
                dictionary=hps,
                convert_bool_to_str=False,
                convert_none_to_str=False,
                force_object_datatype=ensure_or_create(classifier_spec.params_as_object_columns_in_df_search, list),
                check_matching_keys_cols=True,
                check_non_builtin_types=True,
                copy=False
            )
            self.dfs_cv.append(df_cv)

        return agg_loss

    
    def _compute_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.metric == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}.")
        

    def build_df_cv(self, on_empty: Literal["none", "error"] = "error") -> pd.DataFrame | None:
        '''
        Builds the dataframe collecting the executed cv info.
        When these info are missing (empty list) None or an error can be raised.
        '''
        if not self.dfs_cv:
            if on_empty == "none":
                return None
            else:
                raise ValueError("No cv execution info is stored in the instance.")
        return pd.concat([self.dfs_cv], axis=0, ignore_index=True)