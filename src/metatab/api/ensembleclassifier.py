from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.ensemble.single import EnsembleEstimator
from metatab.utils.general import asdict_shallow, ensure_or_create
from metatab.api.metaconfig import MetaConfig
from metatab.classifiers.registry import get_classifier_specs_from_registry

from metatab.utils.api import (
    create_pipeline, 
    encode_y, 
    handle_device, 
    check_validation_set, 
    check_validation_set_classifier_combination
)

if TYPE_CHECKING:
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.utils.types import XType, YType
    from metatab.utils.types import TunableEstimatorType



class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        save_path: str | Path,
        type_classifier: TunableEstimatorType,
        name: str = "ens",
        ensemble_algo = Literal["random", "meta"],
        n_members: int = 16,
        preprocessing: PreprocessingStrategy = "estimator_default",
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

            save_path (str | Path):
                Folder where ensemble members are saved. 
                Members are serialized with pickle and named using the ensemble name plus a number. 
                The folder is created when not existent.
            
            type_classifier (TunableEstimatorType):
                Classifier for which build the ensemble.

            name (str, optional): 
                Name of the ensemble, used as a prefix for member filenames.
            
            ensemble_algo (Literal["random", "meta]):
                Ensemble algo to use:
                - "random": randomly derived configurations are ensembled.
                - "meta": configurations derived from Metatab meta-framework are ensembled.
                In all cases the configurations are ensembled via unweighted prediction averaging.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of hps configuration to derive.

            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy to apply.

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
            self.meta_config.check()

        if self.ensemble_algo == "meta" and self.meta_config is None:
            self.meta_config = MetaConfig(meta_strategy="random_uniform_from_best")

        check_validation_set(validation_set_size)
        
        check_validation_set_classifier_combination(
            validation_set=validation_set_size,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )
        
        resolved_device = handle_device(
            input_device=self.device,
            classifier_spec=classifer_spec,
            type_classifier=self.type_classifier
        )
        
        # execute configuration code necessary for ensembling
        classifer_spec.initialize_search_function()
        label_encoder, y = encode_y(X, y)
        
        resolved_preprocessing = classifer_spec.default_preprocessing \
            if self.preprocessing == "estimator_default" \
            else self.preprocessing
        
        pipe = create_pipeline(
            classifier_class=classifer_spec.classifier_class,
            classifier_params=classifer_spec.fixed_params,
            callbacks_on_classifier_params=classifer_spec.callbacks_on_params,
            y=y,
            preprocessing=resolved_preprocessing,
            classifier_random_state_parameter=classifer_spec.random_state_parameter,
            classifier_nthreads_paramater=classifer_spec.n_threads_parameter,
            classifier_device_parameter=classifer_spec.device_parameter,
            seed=self.seed,
            n_threads=self.n_threads,
            device=resolved_device
        )

        estimator = EnsembleEstimator(
            name=self.name,
            algo=self.ensemble_algo,
            n_members=self.n_members,
            save_path=self.save_path,
            pipe=pipe,
            type_estimator=self.type_classifier,##refactor: change name parameter here
            preprocessing=resolved_preprocessing,
            early_stop_on_validation_set=classifer_spec.early_stop_on_validation_set,
            validation_set_size=validation_set_size,
            sampler_function=classifer_spec.sampler_function,
            time_limit=self.time_limit,
            log=self.log,
            raise_error_fit_member=self.raise_error_fit_member,
            raise_error_void_ensemble=self.raise_error_void_ensemble,
            **ensure_or_create(asdict_shallow(self.meta_config), dict)##refactor here pass directly the metaconfig
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