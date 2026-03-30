from __future__ import annotations

import sys
import time
import warnings
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold
from metatab.utils.exceptions import TimeLimitError
from metatab.utils.core import fit_using_validation_set
from metatab.utils.logging import create_logger
from metatab.utils.pipeline import build_pipeline

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.utils.types import XType, YType
    from metatab.search.configuration import PipelineConfiguration



class EnsembleEstimator:
    '''
    Ensemble PipelineConfigurations using an uniform (unweighted) strategy.

    Parameters:
        name (str): 
            Name of the ensemble that is used as root for members name.

        save_path (None | str | Path):
            Path of the folder where the ensemble members are saved.
            The folder is created when not existent.
            If None the models are kept in memory.

        pipe_configurations (list[PipelineConfigurations]):
            List of pipeline configurations to ensemble.
        
        validation_set_size (float, optional):
            The ratio of validation set draw from train data.
            Ignored by classifiers that does not rely on it and when `n_bag_cv_folds`is specificied.
            In this case the out-of-fold (oof) fraction of the data is used as validation.

        raise_error_fit_member (bool, optional):
            Whether to ignore fit errors in the ensemble building process.
            If True the process is blocked when an fit error is encountered.
            Note that time limit errors are considered as fit errors here.
            If False the process goes on.

        raise_error_void_ensemble (bool, optional):
            Whether to raise an error when a void ensemble is obtained.
            This can be due to time limit or fit failings.
        
        seed (int, optional):
            Seed used to: 
            - derive the validation sets when needed.
            - set the seeds of the classifier to fit.
            - in cv-bagging to get folds.

        time_limit (int, optional):
            Time in seconds to spend for ensemble construction.
            The ensemble building process is stopped when this limit is violated.
            The default is 10 million equal to 115 days approximately, meaning no limit.

        log (int, optional):
            Level of the internal logger. The default assures logs at info level.
            To suppress it set a value of 40 or 50.

    ## Attributes:
        is_void_ (bool): Flag informing whether the ensemble is void.
        fit_time_ (float): Ensemble total fit time in seconds.
        successful_members_ (list[str]): List with the names of the successfully fitted members.
        failed_members_ (list[str]): List with the names of the members which fit process failed.
        successful_confs_ (list[dict]): List of the hps configurations of the successfull members.
        failed_confs_ (list[dict]): List of the hps configurations of the failed members. 
        df_members_ (pd.DataFrame): DataFrame with info about the members fit process.
    '''
    def __init__(
        self,
        name: str,
        save_path: None | str | Path,
        pipe_configurations: list[PipelineConfiguration],
        n_bag_cv_folds: int | None,
        validation_set_size: float,
        raise_error_fit_member: bool,
        raise_error_void_ensemble: bool,
        seed: int,
        device: Literal["cpu", "cuda", "auto"], ## admitting auto here??
        n_threads: int,
        time_limit: int,
        log: int,
    ):
        self.name=name
        self.save_path=save_path
        self.pipe_configurations=pipe_configurations
        self.n_bag_cv_folds=n_bag_cv_folds
        self.validation_set_size=validation_set_size
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.seed=seed
        self.device=device
        self.n_threads=n_threads
        self.time_limit=time_limit
        self.log=log
        

    def fit(self, X: XType, y: YType) -> "EnsembleEstimator":
        start_time = time.time()
        logger = create_logger(self.name, sys.stdout, formatter="standard")
        logger.info(f"Starting prepration phase for '{self.name}' ensemble.")
        
        self._n_members = len(self.pipe_configurations)
        ### REFACTOR: check that all objects are PipelineConfigurations ???

        if self.save_path is not None:
            self._save_path = self.save_path if isinstance(self.save_path, Path) else Path(self.save_path)
            self._save_path.mkdir(parents=True, exist_ok=True)
        else:
            self._save_path = None
        
        rng_ensemble = np.random.default_rng(self.seed)
        member_names = [self.name + "_m" + str(i) for i in range(self._n_members)]
        member_data_indexes = self._get_training_indexes(X, y)
        time_preparation = round((time.time() - start_time)/60, 2)
        logger.info(f"Preparation phase completed in {time_preparation} minutes.")
        
        self.failed_members_ = []
        self.successful_members_ = []
        self.failed_confs_ = []
        self.successful_confs_ = []
        self._recap_members: list[dict] = []
        self._is_disk_cleaned = False
        self._has_completed = False
        self._models: list[Pipeline] = []


        for pipe_conf, member_name, (member_train_indexes, member_val_indexes) in zip(
            self.pipe_configurations, 
            member_names, 
            member_data_indexes
        ):
            ### REFACTOR: we do not control for specific errors vs tuning. 
            # An option would be explicetely pass the error that you want to tollerate
            try:
                if self._is_time_limit_violated(start_time):
                    raise TimeLimitError("Violated time limit")
                
                # we do not uniform the original data type to preserve the column check at inference  
                X_train = X.iloc[member_train_indexes, :] if isinstance(X, pd.DataFrame) else X[member_train_indexes, :]
                y_train = y.iloc[member_train_indexes] if isinstance(y, pd.Series) else y[member_train_indexes]
                
                pipe = build_pipeline(
                    preprocessing=pipe_conf.preprocessing,
                    # adding fixed to dynamic hps
                    hps={**pipe_conf.hps, **pipe_conf.classifier_spec.fixed_params}, 
                    classifier_spec=pipe_conf.classifier_spec, 
                    classifier_seed=int(rng_ensemble.integers(0, 2**32)),
                    classifier_device=self.device,
                    classifier_nthreads=self.n_threads,
                    y=y_train
                )

                if pipe_conf.classifier_spec.early_stop_on_validation_set:
                    # use oof as validation when available otherwise use the validation ratio
                    if self.n_bag_cv_folds:
                        X_val = X.iloc[member_val_indexes, :] if isinstance(X, pd.DataFrame) else X[member_val_indexes, :]
                        y_val = y.iloc[member_val_indexes] if isinstance(y, pd.Series) else y[member_val_indexes]
                        validation_set_size = None
                    else:
                        X_val = None
                        y_val = None
                        validation_set_size = self.validation_set_size

                    pipe, fit_time = fit_using_validation_set(
                        pipe=pipe,
                        X=X_train,
                        y=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        validation_set_size=validation_set_size,
                        seed=rng_ensemble.integers(low=0, high=2**30),
                        return_fit_time=True
                    )

                else:
                    t = time.time()
                    pipe.fit(X_train, y_train)
                    fit_time = time.time() - t

                logger.debug(f"'{member_name}' member has been fitted in {round(fit_time / 60, 2)} minutes.")                
                
                # store model
                if self._save_path:
                    with open(self._save_path / f"{member_name}.pkl", "wb") as f:
                        pickle.dump(pipe, f)
                    logger.debug(f"'{member_name}' member saved on disk.")
                else:
                    self._models.append(pipe)

                # the model is considered successful if fitted AND stored
                self.successful_members_.append(member_name)
                self.successful_confs_.append(pipe_conf)

                self._recap_members.append({
                    "member": member_name,
                    "fit_successful": True,
                    "fit_time": fit_time,
                    "error": None
                })

            except Exception as e:
                if self.raise_error_fit_member:
                    raise ValueError(
                        f"The fit or saving process of the '{member_name}' member has failed."
                    ) from e
                
                self.failed_members_.append(member_name)
                self.failed_confs_.append(pipe_conf)

                self._recap_members.append({
                    "member": member_name,
                    "fit_successful": False,
                    "fit_time": np.nan,
                    "error": str(e)
                })

                logger.debug(f"'{member_name}' member fit or saving process has failed.")
        

        self.is_void_ = False if self.successful_members_ else True
        
        if self.is_void_ and self.raise_error_void_ensemble:
            raise ValueError("The ensemble is void. All members fit/saving process failed.")
        
        self.df_members_ = pd.DataFrame(self._recap_members)
        self.fit_time_ = time.time() - start_time
        self._has_completed = True
        
        logger.info(
            f"Ensemble constructed in {round(self.fit_time_ / 60, 2)} minutes" + 
            f" with {len(self.successful_members_)}/{self._n_members} successful fitted members.\n"
        )

        return self


    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict the sample labels. 
        The labels are inferred on the averaged probabilities.
        '''
        self._check_on_predict_calls()
        return np.argmax(self.predict_proba(X), axis=1)
    

    def predict_proba(self, X: XType) -> np.ndarray:
        self._check_on_predict_calls()
        predictions = self._get_members_predicted_probabilities(X)
        return np.stack(predictions, axis=0).mean(axis=0)


    def get_members_predicted_probabilities(self, X: XType) -> dict[str, np.ndarray]:
        '''
        Get the predicted probabilities of the individual ensemble members.
        Returns a dict of member name - predictions couples.
        '''
        self._check_on_predict_calls()
        predictions = self._get_members_predicted_probabilities(X)
        return {k:v for k, v in zip(self.successful_members_, predictions)}
    

    def _get_members_predicted_probabilities(self, X: XType) -> list[np.ndarray]:
        '''
        Get the predicted probabilities of the ensemble members in a list
        that reflects the order of `self.successful_members_`.
        '''
        predictions = []
        if self._save_path:
            for member in self.successful_members_:
                with open(self._save_path / f"{member}.pkl", "rb") as f:
                    model: Pipeline = pickle.load(f) 
                predictions.append(model.predict_proba(X))
        else:
            for model in self._models:
                predictions.append(model.predict_proba(X))
        return predictions


    def delete_models_from_disk(self) -> None:
        '''
        Delete the ensemble models from disk.
        Works also on partally fitted ensemble, 
        i.e ensembles whose fit process has not completed.
        '''
        # check on successful members to allow functionality on arrested ensemble 
        check_is_fitted(self, "successful_members_")

        if self._save_path is None:
            warnings.warn("No path info stored. Models are not on disk.")
            return None

        if self._is_disk_cleaned:
            return None
        
        if hasattr(self, "is_void_") and self.is_void_:
            warnings.warn("The ensemble is void. No model to delete.")
            return None

        for member in self.successful_members_:
            file_model = self._save_path / f"{member}.pkl"
            file_model.unlink(missing_ok=True)

        self._is_disk_cleaned = True


    def collect_fit_info(self) -> dict:
        self._check_completed_ensemble()
        return {
            "is_void_": self.is_void_,
            "fit_time_": self.fit_time_,
            "successful_members_": self.successful_members_, 
            "failed_members_": self.failed_members_,
            "successful_confs_": self.successful_confs_,
            "failed_confs_": self.failed_confs_,
            "df_members_": self.df_members_
        }


    def _check_on_predict_calls(self) -> None:
        self._check_completed_ensemble()
        self._check_void_ensemble()
        self._check_disk_models()


    def _check_void_ensemble(self) -> None:
        if self.is_void_:
            raise ValueError("The ensemble is void.")
    

    def _check_disk_models(self) -> None:
        if self._save_path and self._is_disk_cleaned:
            raise ValueError("The ensemble models have been deleted.")


    def _check_completed_ensemble(self) -> None:
        if not self._has_completed:
            raise ValueError("The ensemble is incomplete, i.e. its building process was arrested.")
        

    def _is_time_limit_violated(self, start_time: float) -> bool:
        return (time.time() - start_time) > self.time_limit
    

    def _get_training_indexes(self, X: XType, y: YType) -> list[tuple[np.ndarray, np.ndarray | None]]:
        '''
        Returns a list of in- and out- of fold indexes as numpy arrays.
        In case of no bagging we return in-fold indexes convering the whole data and None as out-fold.
        '''
        if self.n_bag_cv_folds is None:
            full_data_indexes = np.arange(X.shape[0])
            return [(full_data_indexes, None) for _ in range(self._n_members)]
        else:
            n_repeats = math.ceil(self._n_members / self.n_bag_cv_folds)
            skf = RepeatedStratifiedKFold(
                n_splits=self.n_bag_cv_folds, 
                n_repeats=n_repeats, 
                random_state=self.seed
            )
            return list(skf.split(X, y))
