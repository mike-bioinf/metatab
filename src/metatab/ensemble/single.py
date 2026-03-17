from __future__ import annotations

import sys
import time
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Literal, TYPE_CHECKING, Callable
from functools import partial
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold
from metatab.utils.exceptions import TimeLimitError
from metatab.utils.core import fit_with_early_stop_on_validation_set, set_params_into_clf
from metatab.utils.general import ensure_or_create
from metatab.utils.logging import create_logger
from metatab.metalearning.acquisition_funcs import compute_upper_confidence_bound
from metatab.metalearning.utils import get_estimator_n_candidate_points
from metatab.metalearning.sampler import WrapperRandomSampler
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.metadata_evaluator import MetadataEvaluator
from metatab.metalearning.load import query_surrogate_framework
from metatab.api.metaconfig import MetaConfig
from metatab.ensemble.utils import BagCV

if TYPE_CHECKING:
    from optuna.trial import Trial
    from sklearn.pipeline import Pipeline
    from metatab.utils.types import TunableClassifierType, XType, YType
    from metatab.preprocessing.types import ResolvedPreprocessingStrategy
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams



class EnsembleEstimator:
    '''
    Class that implements inner-estimator ensemble with a fixed preprocessing.

    Key features:
    - Allows to select estimator hps configuration using the 'random' or 'meta' algo.
    - Allows for a meta-driven ensemble where the hps configurations are suggested by a surrogate model.
    - Supports early stop on validation set for the estimators implementing "eval_set-like" interface.
    - Allows to terminate early the ensemble building process based on a temporal limit.

    Parameters
        name (str): 
            Name of the ensemble that is used as root for members name.
        
        algo (Literal["random", "meta"]):
            How to get the hps configurations for the ensemble.

        n_members (int):
            Number of ensemble members.
            In other terms the number of hps configuration to derive.

        save_path (None | str | Path):
            Path of the folder where the ensemble members are saved.
            The folder is created when not existent.
            If None the models are kept in memory.

        pipe (Pipeline):
            Pipeline object headed with a classifier.
        
        type_estimator (TunableClassifierType):
            String estimator type. 
            Info needed in meta-optimization (`meta` algo).
            
        preprocessing (ResolvedPreprocessingStrategy):
            Type of preprocessing used for the pipe object.
            Info needed in meta-optimization (`meta` algo).

        sampler_function (Callable[[Trail], float]): 
            Optuna sampling function that carries the search space.

        fit_classifier_kwargs (None | dict, optional):
            A dict unpackaged in the classifier fit call.
            If None (default) an empty dict is created.
            The dict keys must be already adapted to the pipeline if any.

        early_stop_on_validation_set (bool):
            Whether to early stop on validation set(s).

        eval_set_parameter (str | None, optional):
            Name of the eval_set parameter, i.e. the parameter taking the 
            validation set(s) at fit level. Can be None.
            Ignored when "early_stop_on_validation_set" is False.
        
        validation_set_size (float, optional):
            The ratio of the early stop validation set.
            Ignored when "early_stop_on_validation_set" is False.

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

        meta_features (None | dict, optional):
            It's possible to pre-specify the metafeatures of the fit data,
            and bypass their computation. This is useful in hierarchical ensemble
            since the same metafeatures can be used for multiple estimator-level ensemble.
            Ignored when `algo` is not "meta".

        meta_candidate_points (None | list[dict]):
            It's possible to pre-specify the candidate points to be evaluated by the
            surrogate model in meta-esembling. This is useful in hierarchical ensemble
            since the same points can be used for multiple estimator-level ensemble.
            Ignored when `algo` is not "meta".
        
        raise_error_fit_member (bool, optional):
            Whether to ignore fit errors in the ensemble building process.
            If True the process is blocked when an fit error is encountered.
            Note that time limit errors are considered as fit errors here.
            If False the process goes on.

        raise_error_void_ensemble (bool, optional):
            Whether to raise an error when a void ensemble is obtained.
            This can be due to time limit or fit failings.
        
        seed (int, optional):
            Seed used to derive the hps points in the random building process,
            and to derive the validation sets for the early stopped estimators.

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
        successful_hps_confs_ (list[dict]): List of the hps configurations of the successfull members.
        failed_hps_confs_ (list[dict]): List of the hps configurations of the failed members. 
        df_members_ (pd.DataFrame): DataFrame with info about the members fit process.
    '''
    def __init__(
        self,
        name: str,
        algo: Literal["random", "meta"],
        n_members: int,
        save_path: None | str | Path,
        pipe: Pipeline,
        type_estimator: TunableClassifierType,
        preprocessing: ResolvedPreprocessingStrategy,
        sampler_function: Callable[[Trial], float],
        early_stop_on_validation_set: bool, 
        log: int,
        validation_set_size: float,
        meta_config: None | MetaConfig, ## REFACTOR: adapt class on this
        raise_error_fit_member: bool,
        raise_error_void_ensemble: bool,
        seed: int,
        time_limit: int,
        bag_cv: None | dict | BagCV,
        ## REFACTOR: eliminate these?
        meta_candidate_points: None | list[dict] = None,
        meta_features: None | dict = None,
        fit_classifier_kwargs: None | dict = None,
        eval_set_parameter: str = "eval_set",
    ):
        self.name=name
        self.algo=algo
        self.n_members=n_members
        self.save_path=save_path
        self.pipe=pipe
        self.type_estimator=type_estimator
        self.preprocessing=preprocessing
        self.sampler_function=sampler_function
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs
        self.meta_config = meta_config
        self.meta_features=meta_features
        self.meta_candidate_points=meta_candidate_points
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.log=log
        self.seed=seed
        self.time_limit=time_limit
        self.bag_cv=bag_cv


    ## REFACTOR: consider abstract some logic in utilitites or separate classes 
    # after defining what to do with preprocessings and classifiers extension.
    def fit(self, X: XType, y: YType) -> "EnsembleEstimator":
        start_time = time.time()
        logger = create_logger(self.name, sys.stdout) ## add formatter?
        logger.info(f"Starting prepration phase for '{self.name}' ensemble.")

        if self.save_path is not None:
            self._save_path = self.save_path if isinstance(self.save_path, Path) else Path(self.save_path)
            self._save_path.mkdir(parents=True, exist_ok=True)
        else:
            self._save_path = None

        bag_cv = BagCV.build_from_dict(self.bag_cv) if isinstance(self.bag_cv, dict) else self.bag_cv

        if bag_cv:
            number_bag_iter = bag_cv.n_folds * bag_cv.n_repeats
            if number_bag_iter < self.n_members:
                raise ValueError(
                    f"Number of bag iterations ({number_bag_iter}) is less than members ({self.n_members})."
                )
        
        training_indexes = self._get_training_indexes(X, y, bag_cv)
        fit_classifier_kwargs = ensure_or_create(self.fit_classifier_kwargs, dict)
        hps_confs = self._get_hps_configurations(X, y)
        member_names = [self.name + "_m" + str(i) for i in range(self.n_members)]
        
        if self.early_stop_on_validation_set:
            rng_early_stop = np.random.default_rng(self.seed)
        
        time_preparation = round((time.time() - start_time)/60, 2)
        logger.info(f"Preparation phase completed in {time_preparation} minutes.")
        
        
        self.failed_members_ = []
        self.successful_members_ = []
        self.failed_hps_confs_ = []
        self.successful_hps_confs_ = []
        self._recap_members: list[dict] = []
        self._is_cleaned = False
        self._has_completed = False
        self._models: list[Pipeline] = []


        for hp_conf, member_name, member_train_indexes in zip(hps_confs, member_names, training_indexes):
            try:
                if self._is_time_limit_violated(start_time):
                    raise TimeLimitError("Violated time limit")
                
                # we do not uniform the original data type to preserve the column check at inference  
                X_member = X.iloc[member_train_indexes, :] if isinstance(X, pd.DataFrame) else X[member_train_indexes]
                y_member = y.iloc[member_train_indexes] if isinstance(y, pd.Series) else y[member_train_indexes]

                # deepcopy necessary for catboost cannot be refitted and for storing models in memory 
                pipe = deepcopy(self.pipe)
                set_params_into_clf(pipe, hp_conf)
            
                if self.early_stop_on_validation_set:
                    pipe, fit_time = fit_with_early_stop_on_validation_set(
                        pipe=pipe,
                        X=X_member,
                        y=y_member,
                        seed=rng_early_stop.integers(low=0, high=2**30),
                        eval_set_parameter=self.eval_set_parameter,
                        validation_set_size=self.validation_set_size,
                        fit_classifier_kwargs=fit_classifier_kwargs,
                        return_fit_time=True
                    )
                else:
                    t = time.time()
                    pipe.fit(X_member, y_member, **fit_classifier_kwargs)
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
                self.successful_hps_confs_.append(hp_conf)

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
                self.failed_hps_confs_.append(hp_conf)

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
            f" with {len(self.successful_members_)}/{self.n_members} successful fitted members.\n"
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
        # check on succesful members to allow functionality on arrested ensemble 
        check_is_fitted(self, "successful_members_")

        if self._save_path is None:
            warnings.warn("No path info stored. Models are not on disk.")
            return None

        if self._is_cleaned:
            return None
        
        if hasattr(self, "is_void_") and self.is_void_:
            self._is_cleaned = True
            warnings.warn("The ensemble is void. No model to delete.")
            return None

        for member in self.successful_members_:
            file_model = self._save_path / f"{member}.pkl"
            file_model.unlink(missing_ok=True)

        self._is_cleaned = True


    def collect_fit_info(self) -> dict:
        self._check_completed_ensemble()
        return {
            "is_void_": self.is_void_,
            "fit_time_": self.fit_time_,
            "successful_members_": self.successful_members_, 
            "failed_members_": self.failed_members_,
            "successful_hps_confs_": self.successful_hps_confs_,
            "failed_hps_confs_": self.failed_hps_confs_,
            "df_members_": self.df_members_
        }


    def _check_on_predict_calls(self) -> None:
        self._check_completed_ensemble()
        self._check_void_ensemble()
        self._check_cleaned_ensemble()


    def _check_void_ensemble(self) -> None:
        if self.is_void_:
            raise ValueError("The ensemble is void.")
    

    def _check_cleaned_ensemble(self) -> None:
        if self._is_cleaned:
            raise ValueError("The ensemble models have been deleted.")


    def _check_completed_ensemble(self) -> None:
        if not self._has_completed:
            raise ValueError("The ensemble is incomplete, i.e. its building process was arrested.")
        

    def _is_time_limit_violated(self, start_time: float) -> bool:
        return (time.time() - start_time) > self.time_limit
    

    def _get_training_indexes(self, X: XType, y: YType, bag_cv: None | BagCV) -> list[np.ndarray]:
        '''
        Returns a list of training indexes as numpy arrays.
        In case of bag_cv equal None a list of indexes covering the whole data is returned.
        '''
        if bag_cv is None:
            return [np.arange(X.shape[0])] * self.n_members
        else:
            skf = RepeatedStratifiedKFold(
                n_splits=bag_cv.n_folds, 
                n_repeats=bag_cv.n_repeats, 
                random_state=bag_cv.seed
            )
            return [tuple_index[0] for tuple_index in skf.split(X, y)]
            
    
    def _get_hps_configurations(self, X: XType, y: YType) -> list[dict]:
        sampler = WrapperRandomSampler()
        mfe = CustomMFE()
        
        if self.algo == "random":
            sampler.fit(self.sampler_function, seed=self.seed)    
            points = sampler.sample_points(self.n_members)

        elif self.algo == "meta":
            ## REFACTOR: abstract meta-points generation code 
            if self.meta_features is None:
                metafeatures, _ = mfe.fit(X, y).extract()
            else:
                metafeatures = self.meta_features

            metafeatures["preprocessing"] = self.preprocessing
            
            if self.meta_candidate_points is None:
                n_candidate_points = max(get_estimator_n_candidate_points(self.type_estimator), self.n_members) \
                    if self.meta_strategy_params is None \
                    else self.meta_strategy_params.n_candidate_points
                
                sampler.fit(self.sampler_function, seed=self.meta_seed)
                candidate_points = sampler.sample_points(n_candidate_points)
            else:
                candidate_points = self.meta_candidate_points
            
            df_candidate_points = pd.DataFrame(candidate_points)
        
            # we create a copy since the original df is not optimized in memory due to assign
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)
                metadata = df_candidate_points.assign(**metafeatures).copy()

            acquisition_func = partial(
                compute_upper_confidence_bound,
                k=1,
                mean_direction="lower_is_better", # we currently use only the logloss
            )

            # use the input model or use the default
            surrogate_model = joblib.load(self.meta_surrogate_model) \
                if self.meta_surrogate_model \
                else query_surrogate_framework(self.type_estimator)

            meta_evaluator = MetadataEvaluator(surrogate_model, acquisition_func)
            meta_evaluator.fit(metadata, candidate_points).evaluate_candidates()

            if self.meta_strategy == "best":
                points = meta_evaluator.propose_n_best(n_best=self.n_members)
            
            elif self.meta_strategy == "random_from_best":
                # we use a ratio of 1 to 5 by default when possible
                top = min(self.n_members * 5, n_candidate_points) \
                    if self.meta_strategy_params is None \
                    else self.meta_strategy_params.top
                
                # we use the instance seed to allow variability when not hardcoded in the supplied params
                points = meta_evaluator.propose_random_from_top(
                    n_proposed=self.n_members,
                    top=top,
                    seed=self.seed if self.meta_strategy_params is None else self.meta_strategy_params.seed
                )
            
            elif self.meta_strategy == "uniform_from_best":
                # we use a step of 10 by default when possible
                if self.meta_strategy_params is None:
                    max_step = int(n_candidate_points / self.n_members)
                    step_size = 10 if max_step > 10 else max_step
                else:
                    step_size = self.meta_strategy_params.step_size
                
                points = meta_evaluator.propose_uniform_from_top(
                    n_steps=self.n_members,
                    step_size=step_size
                )

            elif self.meta_strategy == "random_uniform_from_best":
                # we use a step of 10 by default when possible
                if self.meta_strategy_params is None:
                    max_step = int(n_candidate_points / self.n_members)
                    step_size = 10 if max_step > 10 else max_step
                else:
                    step_size = self.meta_strategy_params.step_size

                # we use the instance seed to allow variability when not hardcoded in the supplied params
                points = meta_evaluator.propose_random_uniform_from_top(
                    n_steps=self.n_members,
                    step_size=step_size,
                    seed=self.seed if self.meta_strategy_params is None else self.meta_strategy_params.seed
                )
        
        return points