from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.ensemble import EnsembleEstimator
from metatab.api.metaconfig import MetaConfig
from metatab.classifiers.registry import get_classifier_specs_from_registry
from metatab.search.search import UnoptimizedRandomSearch

from metatab.utils.api import ( 
    encode_y, 
    handle_device, 
    check_validation_set, 
    check_validation_set_classifier_combination
)

if TYPE_CHECKING:
    from metatab.search.configuration import PipelineConfiguration
    from metatab.preprocessing import PreprocessingStrategy
    from metatab.utils.types import XType, YType
    from metatab.utils.types import TunableClassifierType



class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        type_classifier: TunableClassifierType,
        save_path: None | str | Path,
        name: str = "ens",
        ensemble_algo = Literal["random", "meta"],
        n_members: int = 16,
        preprocessing: PreprocessingStrategy | list[PreprocessingStrategy] | list[list[PreprocessingStrategy]] = "zero_variance",
        n_bag_cv_folds: int | None = None,
        time_limit: int = 10_000_000,
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        raise_error_fit_member: bool = False,
        raise_error_void_ensemble: bool = True,
        meta_config: None | MetaConfig = None,
        log: int = 20
    ):
        '''
        Ensemble a classifier.

        Parameters:
            type_classifier (TunableClassifierType):
                Classifier to ensemble.

            save_path (str | Path):
                Disk folder where to save ensemble models. Models are serialized with pickle. 
                The folder is created when not existent. If None models are kept in memory.
            
            name (str, optional): 
                Name of the ensemble, used as a prefix for member filenames.
            
            ensemble_algo (Literal["random", "meta]):
                Ensemble algo to use:
                - "random": randomly derived configurations are ensembled.
                - "meta": configurations derived from Metatab meta-framework are ensembled.
                In all cases the configurations are ensembled via unweighted prediction averaging.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of configurations to derive.

            preprocessing (PreprocessingStrategy | list[PreprocessingStrategy] | list[list[PreprocessingStrategy]], optional):
                Preprocessing strategies to use.
                If a single strategy is passed then it is used in all members.
                If multiple strategies are passed then a random selection is done on them.
                Is possible to specify strategies composed by multiple steps to apply together following the input order. 
                So for example ["log", "clr"] signal to use two single options.
                Instead [["log", "clr"]] signal to use a single preprocessing composed by 2 step.
                Is also possible to specify multiple multi-steps preprocessing.
                Custom preprocessing cannot currently be specified.

            n_bag_cv_folds (int | None, optional):
                Number of folds to use in a cv bagging procedure.
                ### REFACTOR: explain better.

            time_limit (int, optional):
                Time in seconds to spend for ensemble construction.
                The ensemble building process is stopped when this limit is violated.
                The default is 10 million equal to 115 days approximately, meaning no limit.

            seed (int, optional):
                Seed controlling the randomness inherent to the classifier and validation splits when used.

            n_threads (int, optional):
                Number of threads used to parallelize the classifiers fitting process.
                Note that the parallelization happens only in the member fitting process
                and not at ensemble level.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device to fit the model(s) on.
                - "cpu" or "cuda" explicitly selects the device.
                - "auto" uses "cuda" if available and supported by the classifier; otherwise "cpu".

            raise_error_fit_member (bool, optional):
                Whether to stop the process when a member fails to fit. 
                Time-limit errors are also considered fit errors. 
                If False, the process continues despite failures.

            raise_error_void_ensemble (bool, optional):
                Whether to raise an error when the fit process fails for all members.

            meta_config(None | MetaConfig, optional):
                Config class for the "meta" algorithm.
                Expands to the default configuration when None and `tune_algo` equal "meta".
                Must be None when `tune_algo` != "meta", otherwise an error is raised.

            log (int, optional):
                Logging level. 
                Default provides info-level logs. 
                Set to 40 or 50 to suppress logging.

                
        ## Attributes:

            is_void_ (bool): 
                Flag informing whether the ensemble is void.
            
            fit_time_ (float): 
                Ensemble total fit time in seconds.
            
            successful_members_ (list[str]): 
                List with the names of the successfully fitted members.
            
            failed_members_ (list[str]): 
                List with the names of the members which fit process failed.
            
            successful_hps_confs_ (list[dict]): 
                List of the hps configurations of the successful members.
            
            failed_hps_confs_ (list[dict]): 
                List of the hps configurations of the failed members. 
            
            df_members_ (pd.DataFrame): 
                DataFrame summarizing members fit process info.
            
            classes_ (np.ndarray): 
                The array of class labels learnt at fit time.
        '''
        self.save_path=save_path
        self.type_classifier=type_classifier
        self.name=name
        self.ensemble_algo=ensemble_algo
        self.n_members=n_members
        self.preprocessing=preprocessing
        self.n_bag_cv_folds=n_bag_cv_folds
        self.time_limit=time_limit
        self.log=log
        self.seed=seed
        self.n_threads=n_threads
        self.device=device
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.meta_config=meta_config


    def fit(
        self,
        X: XType,
        y: YType,
        validation_set_size: float | None = None
    ):
        '''
        ## REFACTOR: add documentation here.
        '''
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        classifer_spec = get_classifier_specs_from_registry(self.type_classifier)

        if self.ensemble_algo not in ["random", "meta"]:
            raise ValueError("'ensemble_algo' must be equal to 'random' or 'meta'.")
        
        if self.ensemble_algo != "meta" and self.meta_config:
            raise ValueError(f"With 'ensemble_algo' != 'meta', 'meta_config' must be None.")
        
        if self.ensemble_algo == "meta" and self.meta_config is not None:
            self.meta_config._check()

        if self.ensemble_algo == "meta" and self.meta_config is None:
            self.meta_config = MetaConfig(meta_strategy="random_uniform_from_best")
        
        if self.n_members < 1:
            raise ValueError("'n_members' must be greater than 0.")
        
        if self.n_bag_cv_folds and self.n_bag_cv_folds < 2:
            raise ValueError("'n_bag_cv_folds' must be an int greater or equal than 2.")

        check_validation_set(validation_set_size)
        
        check_validation_set_classifier_combination(
            validation_set=validation_set_size,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )
        
        ### REFACTOR: render this more clear for users if maintened this way
        if validation_set_size and self.n_bag_cv_folds:
            raise ValueError(
                "'validation_set_size' must be None when 'n_bag_cv_folds' is specified."
            )
        
        resolved_device = handle_device(
            input_device=self.device,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )
        
        # execute configuration code necessary for ensembling
        classifer_spec.initialize_search_function()
        label_encoder, y = encode_y(X, y)                
        pipe_confs = self._get_pipe_configurations(classifer_spec)

        estimator = EnsembleEstimator(
            name=self.name,
            save_path=self.save_path,
            pipe_configurations=pipe_confs,
            n_bag_cv_folds=self.n_bag_cv_folds,
            validation_set_size=validation_set_size,
            raise_error_fit_member=self.raise_error_fit_member,
            raise_error_void_ensemble=self.raise_error_void_ensemble,
            seed=self.seed,
            device=resolved_device,
            n_threads=self.n_threads,
            time_limit=self.time_limit,
            log=self.log
        )

        self.estimator_ = estimator.fit(X, y)     
        self.classes_ = label_encoder.classes_
        for k, v in self.estimator_.collect_fit_info().items(): setattr(self, k, v)
        return self
    

    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict class for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            np.ndarray: The predicted classes.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.
        
        Returns:
            np.ndarray: The class probabilities of the input samples.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)
    

    def get_members_predicted_probabilities(self, X: XType) -> dict[str, np.ndarray]:
        '''
        Get the class probabilities of every ensemble member.

        Parameters:
            X (XType): Input samples.

        Returns:
            dict[str,np.ndarray]: 
            The dict with member names as keys and predicted probabilities as values.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.get_members_predicted_probabilities(X)
    
    
    def delete_models_from_disk(self) -> None:
        '''Delete the ensemble models from disk'''
        check_is_fitted(self, "estimator_")       
        self.estimator_.delete_models_from_disk()


    def _get_pipe_configurations(self, classifier_spec) -> list[PipelineConfiguration]:
        if self.ensemble_algo == "random":
            search_object = UnoptimizedRandomSearch(
                classifier_spec=classifier_spec,
                preprocessing=self.preprocessing,
                n_confs=self.n_members,
                seed=self.seed
            )
        else:
            ## REFACTOR: build MetaSearch object
            pass

        return search_object.get_configurations()