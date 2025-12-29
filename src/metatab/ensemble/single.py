from __future__ import annotations

import sys
import time
import warnings
import joblib
import pickle
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from functools import partial
from sklearn.utils.validation import check_is_fitted
from metatab.metatab_utils.exceptions import TimiLimitError
from metatab.estimators.utils.general import collect_sklearn_classification_fit_info_from_data
from metatab.estimators.utils.fit import fit_with_early_stop_on_validation_set, set_params_into_clf
from metatab.metalearning.acquisition_funcs import compute_upper_confidence_bound
from metatab.metalearning.utils import check_meta_strategy, check_meta_strategy_params, get_estimator_n_candidate_points
from metatab.metalearning.sampler import HyperoptRandomSampler
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.metadata_evaluator import MetadataEvaluator
from metatab.metalearning.load import query_surrogate_framework
from metatab.metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.estimators.utils.types import Classifier, TunableEstimatorType
    from metatab.preprocessing.types import ResolvedPreprocessingStrategy
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.metatab_utils.types import XType, YType



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

        save_path (str | Path):
            Path of the folder where the ensemble members are saved.
            These will be serialized with the pickle module in files
            named after the members (name ensemble + number member).
            Note that the folder is created if not existent.

        clf_or_pipe (Classifier | Pipeline):
            Classifier or Pipeline object with a classifier as head to ensemble.
        
        type_estimator (TunableEstimatorType):
            String estimator type. 
            Info needed in meta-optimization (`meta` algo).
            
        clf_or_pipe_preprocessing (ResolvedPreprocessingStrategy):
            Type of preprocessing used for the clf_or_pipe object.
            Info needed in meta-optimization (`meta` algo).

        params_distributions (dict): 
            Classifier search space from which the hps configuration are taken.

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

    ** Attributes:
        is_void_ (bool): Flag informing whether the ensemble is void.
        fit_time_ (float): Ensemble total fit time in seconds.
        successful_members_ (list[str]): List with the names of the successfully fitted members.
        failed_members_ (list[str]): List with the names of the members which fit process failed.
        successful_hps_confs_ (list[dict]): List of the hps configurations of the successfull members.
        failed_hps_confs_ (list[dict]): List of the hps configurations of the failed members. 
        df_members_ (pd.DataFrame): DataFrame with info about the members fit process.
        classes_ (np.ndarray): Array of unique classes seen at fit level.
        n_features_in_ (int): Number of features seen at fit level.
        is_cleaned_ (bool): 
            Flag informing whether the ensemble models have been deleted from disk using the
            "delete_models_from_disk" method.
        feature_names_in_ (np.ndarray): 
            Names of the features seen at fit level.
            This attribute exists only when the instance is fitted with pandas dataframe 
            with all string columns.
    '''
    def __init__(
        self,
        name: str,
        algo: Literal["random", "meta"],
        n_members: int,
        save_path: str | Path,
        clf_or_pipe: Classifier | Pipeline,
        type_estimator: TunableEstimatorType,
        clf_or_pipe_preprocessing: ResolvedPreprocessingStrategy,
        params_distributions: dict,
        early_stop_on_validation_set: bool, 
        eval_set_parameter: str = "eval_set",
        validation_set_size: float = 0.3,
        fit_classifier_kwargs: None | dict = None,
        meta_surrogate_model: None | str | Path = None,
        meta_strategy: MetaStrategy = "random_uniform_from_best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_seed: int = 42,
        meta_features: None | dict = None,
        meta_candidate_points: None | list[dict] = None,
        raise_error_fit_member: bool = False,
        raise_error_void_ensemble: bool = True,
        seed: int = 0,
        time_limit: int = 10_000_000,
        log: int = 20
    ):
        self.name=name
        self.algo=algo
        self.n_members=n_members
        self.save_path=save_path
        self.clf_or_pipe=clf_or_pipe
        self.type_estimator=type_estimator
        self.clf_or_pipe_preprocessing=clf_or_pipe_preprocessing
        self.params_distributions=params_distributions
        self.early_stop_on_validation_set=early_stop_on_validation_set
        self.eval_set_parameter=eval_set_parameter
        self.validation_set_size=validation_set_size
        self.fit_classifier_kwargs=fit_classifier_kwargs
        self.meta_surrogate_model=meta_surrogate_model
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_seed=meta_seed
        self.meta_features=meta_features
        self.meta_candidate_points=meta_candidate_points
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.log=log
        self.seed=seed
        self.time_limit=time_limit


    def fit(self, X: XType, y: YType) -> "EnsembleEstimator":
        start_time = time.time()

        if self.algo not in ["random", "meta"]:
            raise ValueError("algo must be equal to 'random' or 'meta'.")

        if self.algo == "meta":
            check_meta_strategy(self.meta_strategy)
            check_meta_strategy_params(self.meta_strategy, self.meta_strategy_params, safe_none_params=True)
        
        logger = self._get_logger()
        logger.info(f"Starting building process of '{self.name}' ensemble.")

        self._save_path = self.save_path if isinstance(self.save_path, Path) else Path(self.save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)
        
        fit_classifier_kwargs = ensure_or_create(self.fit_classifier_kwargs, dict)
        hps_confs = self._get_hps_configurations(X, y)
        member_names = [self.name + "_m" + str(i) for i in range(self.n_members)]
        
        time_prepation = round((time.time() - start_time)/60, 2)
        logger.info(f"Obtained hps configurations using the {self.algo} algo in {time_prepation} minutes.")
        
        if self.early_stop_on_validation_set:
            rng_early_stop = np.random.default_rng(self.seed)

        self.failed_members_ = []
        self.successful_members_ = []
        self.failed_hps_confs_ = []
        self.successful_hps_confs_ = []
        self._recap_members: list[dict] = []
        self.is_cleaned_ = False

        for hp_conf, member_name in zip(hps_confs, member_names):
            try:
                if self._is_time_limit_violated(start_time):
                    raise TimiLimitError("Violated time limit")
                
                # deepcopy necessary since catboost cannot be refitted
                clf_or_pipe = deepcopy(self.clf_or_pipe)
                set_params_into_clf(clf_or_pipe, hp_conf)
            
                if self.early_stop_on_validation_set:
                    clf_or_pipe, fit_time = fit_with_early_stop_on_validation_set(
                        clf_or_pipe=clf_or_pipe,
                        X=X,
                        y=y,
                        seed=rng_early_stop.integers(low=0, high=2**30),
                        eval_set_parameter=self.eval_set_parameter,
                        validation_set_size=self.validation_set_size,
                        fit_classifier_kwargs=fit_classifier_kwargs,
                        return_fit_time=True
                    )
                else:
                    t = time.time()
                    clf_or_pipe.fit(X, y, **fit_classifier_kwargs)
                    fit_time = time.time() - t

                logger.debug(f"'{member_name}' member has been fitted in {round(fit_time / 60, 2)} minutes.")                
                
                # save classifier
                with open(self._save_path / f"{member_name}.pkl", "wb") as f:
                    pickle.dump(clf_or_pipe, f)

                logger.debug(f"'{member_name}' member saved on disk.")
                
                # the model is considered successful if fitted AND saved on disk
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
                        f"The fit or saving process of the {member_name} ensemble member has failed" +
                        f" with the following error: {e}"
                    )
                
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
        
        for k, v in collect_sklearn_classification_fit_info_from_data(X, y).items():
            setattr(self, k, v)
        
        self.fit_time_ = time.time() - start_time
        
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
        for member in self.successful_members_:
            path_successful_member = self._save_path / f"{member}.pkl"
            member_model: Classifier | Pipeline = self._try_load_model(path_successful_member)
            predictions.append(member_model.predict_proba(X))
        return predictions


    # We want only this method to work when we raise errors for single
    # member fails and a fail is encountered for the second to last model.
    def delete_models_from_disk(self) -> None:
        '''
        Delete the ensemble models from disk.
        Works also for partially fitted ensemble, i.e. ensembles whose fit 
        process has been stopped before completion for single model fails.
        '''
        check_is_fitted(self, "successful_members_")

        if self.is_cleaned_:
            return None
        
        if hasattr(self, "is_void_") and self.is_void_:
            self.is_cleaned_ = True
            warnings.warn("The ensemble is void. No model to delete.")
            return None

        for member in self.successful_members_:
            file_model = self._save_path / f"{member}.pkl"
            file_model.unlink(missing_ok=True)

        self.is_cleaned_ = True
        

    def _check_on_predict_calls(self) -> None:
        # we perform the fitted check on "fit_time_" since is defined at the end of the fit call. 
        # This means that "partially fitted" ensemble does not have this attribute. 
        # We do not want to predict with partially fitted ensemble since 
        # one should allow for failings if this behaviour is desired.
        check_is_fitted(self, "fit_time_")
        self._check_void_ensemble()
        self._check_cleaned_ensemble()


    def _check_void_ensemble(self) -> None:
        if self.is_void_:
            raise ValueError("The ensemble is void.")
    

    def _check_cleaned_ensemble(self) -> None:
        if self.is_cleaned_:
            raise ValueError("The ensemble models have been deleted.")


    @staticmethod
    def _try_load_model(path: str | Path) -> Classifier | Pipeline:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f"Error encountered in the loading process of '{path}': {e}"
            )
        return model


    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.log)
            logger.addHandler(handler)
        logger.propagate = False
        return logger


    def _is_time_limit_violated(self, start_time: float) -> bool:
        return (time.time() - start_time) > self.time_limit
        

    def _get_hps_configurations(self, X: XType, y: YType) -> list[dict]:
        sampler = HyperoptRandomSampler()
        mfe = CustomMFE()
        
        if self.algo == "random":
            sampler.fit(self.params_distributions, seed=self.seed)    
            points = sampler.sample_points(self.n_members)

        elif self.algo == "meta":
            if self.meta_features is None:
                metafeatures, _ = mfe.fit(X, y).extract()
            else:
                metafeatures = self.meta_features

            metafeatures["preprocessing"] = self.clf_or_pipe_preprocessing
            
            if self.meta_candidate_points is None:
                n_candidate_points = max(get_estimator_n_candidate_points(self.type_estimator), self.n_members) \
                    if self.meta_strategy_params is None \
                    else self.meta_strategy_params.n_candidate_points
                
                sampler.fit(self.params_distributions, seed=self.meta_seed)
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