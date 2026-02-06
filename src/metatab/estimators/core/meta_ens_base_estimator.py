from __future__ import annotations

import pickle
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Callable
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.estimators.utils.general import learn_sklearn_features_attributes, check_predict_features
from metatab.ensemble.single import EnsembleEstimator
from metatab.estimators.core.utils import handle_device, create_pipeline, encode_y, check_validation_set
from metatab.preprocessing.utils import resolve_preprocessing_info
from metatab.preprocessing.collect import collect_fit_preprocessing_info
from metatab.estimators.params.utils import pick_estimator_tune_space
from metatab.metalearning.utils import check_meta_strategy_params

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from metatab.estimators.utils.types import Classifier, EstimatorType
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.metatab_utils.types import XType, YType



class BaseEnsembleEstimator(ClassifierMixin, BaseEstimator):
    '''
    Base class for ensemble estimators. Provides:
    - fit capabilities
    - predict capabilities
    - serialization funtionality
    - disk model deletion method
    - collection of data preprocessing info
    '''
    if TYPE_CHECKING:
        estimator_ : EnsembleEstimator
    
    def fit(
        self,
        X: XType, 
        y: YType,
        type_ensemble: Literal["meta", "random"],
        classifier_class: Classifier,
        classifier_random_state_parameter: str | None,
        classifier_nthreads_paramater: str | None,
        classifier_device_parameter: str | None,
        type_estimator: EstimatorType,
        fixed_params: dict,
        callbacks_on_fixed_params: list[Callable[[dict, pd.Series, bool], dict]] | None = None,
        validation_set: float | tuple[XType, YType] | None = None
    ):
        '''
        Fit the meta-ensemble estimator.
        We pass the specifics of the concrete class along the fit data.

        
        Parameters:
            X (XType): Train data.
            
            y (YType): Train labels.

            type_ensemble (Literal["meta", "random"]):
                Wheter is a meta or random ensemble. So alias for "algo" parameter.

            classifier_class (Classifier): Classifier class.

            classifier_random_state_parameter (str | None):
                Name of the classifier parameter accepting the random state info.
                None is used to signal that the classifier does not accept a random_state-like 
                parameter and therefore the `self.seed` info is unused.
            
            classifier_nthreads_paramater (str | None):
                Name of the classifier parameter accepting the number of threads info to use in fit.
                None is used to signal that the classifier does not accept a n_threads-like 
                parameter and therefore the `self.n_threads` info is unused.

            classifier_device_parameter (str | None):
                Name of the classifier parameter acceting the device info.
                None is used to signal that the classifier does not have a device-like parameter.
                In this case the `self.device` info is unused.

            concrete_estimator_class (Estimator): 
                Ensembled estimator class to instantiate.
            
            type_estimator (EstimatorType):
                String specifyng the estimator base type.
            
            fixed_params (dict):
                Dict of fixed hps values.
            
            callbacks_on_fixed_params (list[Callable[[dict, pd.Series, bool], dict]] | None, optional):
                List of functions to apply to the fixed params before fitting.
                They are applied sequentially following the list order.
                The output of the first is passed in input to the second and so on.
                They must share the same signature (params, y, copy) (this is not checked by the code).
                Pass an empty list or None to skip this functionality.
            
            validation_set (float | tuple[XType, YType] | None, optional):
                Can be either:
                - Float indicating the fraction of train data to use as validation.
                - The X, y validation sets directly.
                - None to signal that the classifier does not need a validation set.

        
        ## Attributes:
            
            List of attributes learned in the fit process.

            is_void_ (bool): 
                Flag informing whether the ensemble is void.
            
            fit_time_ (float): 
                Ensemble total fit time in seconds.
            
            successful_members_ (list[str]): 
                List with the names of the successfully fitted members.
            
            failed_members_ (list[str]): 
                List with the names of the members which fit process failed.
            
            successful_hps_confs_ (list[dict]): 
                List of the hps configurations of the successfull members.
            
            failed_hps_confs_ (list[dict]): 
                List of the hps configurations of the failed members. 
            
            df_members_ (pd.DataFrame): 
                DataFrame with info about the members fit process.
            
            classes_ (np.ndarray): 
                The array of class labels learnt at fit time.
            
            n_features_in_ (int): 
                Number of features seen at fit level.
            
            is_cleaned_ (bool): 
                Flag informing whether the ensemble models have been deleted from disk using the
                "delete_models_from_disk" method.
            
            feature_names_in_ (np.ndarray): 
                Names of the features seen at fit level.
                This attribute exists only when the instance is fitted with pandas dataframe 
                with all string columns.
        
        Returns:
            self
        '''
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        check_validation_set(validation_set)
        resolved_device = handle_device(self.device, type_estimator)
        label_encoder, y = encode_y(X, y)
        
        pipe = create_pipeline(
            classifier_class=classifier_class,
            type_estimator=type_estimator,
            fixed_params=fixed_params,
            callbacks_on_fixed_params=callbacks_on_fixed_params,
            y=y,
            preprocessing=self.preprocessing,
            classifier_random_state_parameter=classifier_random_state_parameter,
            classifier_nthreads_paramater=classifier_nthreads_paramater,
            classifier_device_parameter=classifier_device_parameter,
            seed=self.seed,
            n_threads=self.n_threads,
            resolved_device=resolved_device
        )
    
        ###REFACTOR: are not used???
        # fit_classifier_kwargs = add_prefix_to_params_when_absent(
        #     params_dict=ensure_or_create(fit_classifier_kwargs, dict),
        #     string=f"{pipe.steps[-1][0]}__"
        # )

        # get the "meta" additional parameters conditionally on ensemble type
        if type_ensemble == "random":
            meta_ensemble_parameters = {}
        
        elif type_ensemble == "meta":
            check_meta_strategy_params(
                self.meta_strategy, 
                self.meta_strategy_params, 
                safe_none_params=True
            )
            meta_ensemble_parameters = {
                "meta_strategy": self.meta_strategy,
                "meta_strategy_params": self.meta_strategy_params,
                "meta_surrogate_model": self.meta_surrogate_model,
                "meta_seed": self.meta_seed,
                "meta_features": self.meta_features,
                "meta_candidate_points": self.meta_candidate_points
            }
        
        else:
            raise ValueError("type_ensemble must be 'meta' or 'random'.")
        
        tune_space = self.tune_space \
            if isinstance(self.tune_space, dict) \
            else pick_estimator_tune_space(type_estimator, self.tune_space)
        
        estimator = EnsembleEstimator(
            name=self.name,
            algo=type_ensemble,
            n_members=self.n_members,
            save_path=self.save_path,
            pipe=pipe,
            type_estimator=type_estimator,
            preprocessing=resolve_preprocessing_info(self.preprocessing),
            early_stop_on_validation_set=False if validation_set is None else True,
            validation_set=validation_set,
            fit_classifier_kwargs=None,  ###REFACTOR: IS NOT USED????
            params_distributions=tune_space,
            time_limit=self.time_limit,
            log=self.log,
            raise_error_fit_member=self.raise_error_fit_member,
            raise_error_void_ensemble=self.raise_error_void_ensemble,
            **meta_ensemble_parameters
        )

        self.estimator_ = estimator.fit(X, y)        
        self.classes_ = label_encoder.classes_
        sklearn_info = learn_sklearn_features_attributes(X)
        ensemble_info = self.estimator_.collect_fit_info()
        fit_info = {**sklearn_info, **ensemble_info}
        for k, v in fit_info.items():
            setattr(self, k, v)
        
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
        check_predict_features(self, X)
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
        check_predict_features(self, X)
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
        check_predict_features(self, X)
        return self.estimator_.get_members_predicted_probabilities(X)
    

    def delete_models_from_disk(self) -> None:
        '''Delete the ensemble models from disk.'''
        self.estimator_.delete_models_from_disk()
        self.is_cleaned_ = True


    ##REFACTOR: this must be thought and placed better
    def collect_fit_preprocessing_info(self) -> dict:
        check_is_fitted(self, "estimator_")
        # this check is useful also in this case
        self.estimator_._check_on_predict_calls()
        model = self.estimator_._save_path / f"{self.estimator_.successful_members_[0]}.pkl"
        return collect_fit_preprocessing_info(
            self.estimator_._try_load_model(model), 
            self.preprocessing
        )


    def save(self, filepath: str | Path, check_is_fitted: bool = True) -> None:
        '''
        Serialize the instance using pickle.
        Allows for a conditional check on the "fitted nature" of the estimator.
        '''
        if check_is_fitted and not hasattr(self, "estimator_"):
            raise ValueError("The estimator instance is not fitted (no 'estimator_' attibute).")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        
    def _get_init_configuration(self) -> dict:
        '''
        Get the initialization configuration of the instance.
        We return in a dict all the info necessary to recreate the 
        instance from it's initialization parameters.
        '''
        if hasattr(self, "estimator_"):
            raise ValueError(
                "The ensemble is fitted! The init configuration cannot be safely retrieved."
            )

        params = deepcopy(self.__dict__)

        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "params": params,
        }




class MetaEnsembleInitializer:
    def __init__(
        self,
        save_path: str | Path,
        name: str = "meta_ens",
        n_members: int = 16,
        seed: int = 0,
        time_limit: int = 10_000_000,
        log: int = 20,
        raise_error_fit_member: bool = False,
        raise_error_void_ensemble: bool = True,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        meta_surrogate_model: None | str | Path = None,
        meta_strategy: MetaStrategy = "random_uniform_from_best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_seed: int = 42,
        preprocessing: Literal["estimator_default"] = "estimator_default",
        tune_space: Literal["default"] = "default"
    ):
        '''
        The meta ensembled estimators are backed by a meta-learning framework that suggestes,
        evaluates and selects points drawn from pre-defined tuning spaces.
        In detail the process consists of the following phases:

        1. Candidate points derivation.
        The candidate points are drawn in a fixed number, customizable via `meta_strategy_params`,
        in a reproducible way thanks to the `meta_seed`. We do NOT recommend to change these 
        values, since changing them means obtaining points that do not belong
        to our prior. In other words keeping the default assures to evaluate points
        that have all been tested on prior data.

        2. Promisingness score evaluation.
        The promisingness of the candidate points is evaluated though a surrogate model,
        fitted on our prior optimization-data, and the UCB acquistion function.
        In particular the surrogate model uses also a pre-defined set of metafeatures
        computed on the input data to predict point quality.

        3. Point selection.
        The `n_members` most promising points are then selected using the `meta_strategy`.

        4. Ensembling.
        The selected points are then used to build the ensemble. 

        
        ### User Note:
            We highly suggest to NOT preprocess the microbial profiles apart expressing them in the 
            "relative" format (i.e rows summing to 1). This is because we automatically select the 
            most appropiate preprocessing scheme according to the classifier.
            In addition we learn the data metafeatures before preprocessing. 
            Therefore a custom pre-preprocessing can potentially hurt performance.


        Parameters:

            save_path (str | Path):
                Path of the folder where the ensemble members are saved.
                These will be serialized with the pickle module in files
                named after the members (name ensemble + number member).
                Note that the folder is created if not existent.
            
            name (str, optional): 
                Name of the ensemble that is used as root for members name.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of hps configuration to derive.

            seed (int, optional):
                Seed controlling the randomness inherent to the estimator and
                to the validation sets determination when early stop is enabled.

            time_limit (int, optional):
                Time in seconds to spend for ensemble construction.
                The ensemble building process is stopped when this limit is violated.
                The default is 10 million equal to 115 days approximately, meaning no limit.

            log (int, optional):
                Level of the internal logger. The default assures logs at info level.
                To suppress it set a value of 40 or 50.

            raise_error_fit_member (bool, optional):
                Whether to ignore fit errors in the ensemble building process.
                If True the process is blocked when an fit error is encountered.
                Note that time limit errors are considered as fit errors here.
                If False the process goes on.

            raise_error_void_ensemble (bool, optional):
                Whether to raise an error when a void ensemble is obtained.
                This can be due to time limit or fit failings.

            n_threads (int, optional):
                Number of threads used to parallelize the classifiers fitting process.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device where to fit the model(s). 
                Note that for some estimators cannot be run on "cuda" raising an error.
                If "auto" then it selects cuda if available AND the estimator requires GPU else cpu.

            meta_surrogate_model (None | str | Path, optional):
                Surrogate model to use for point performance prediction.
                If str or Path, then the object pointed by the path is used as surrogate model.
                This must be a joblib serialized object.
                If None the "default" surrogate model is used.

            meta_strategy (MetaStrategy, optional):
                Set the strategy used by the metalearning framework to select points.
                It has no effect when `algo` is not "meta".
                - "best": select the top-n configurations.
                - "random_from_best": random selection from the top.
                - "uniform_from_best": uniform step selection from the top.
                - "random_uniform_from_best": random selection within uniform intervals from the top.

            meta_strategy_params (None | MetaStrategyParams, optional):
                Meta strategy specifics in form of dataclass.
                If None the default specifics are applied.

            meta_seed (int, optional):
                Seed used specifically to draw condidate points in the meta-optimization scenario.
                Importanlty the default value of 42 is the one used to generate the prior.
                Therefore using the default seed allow to draw and evaluate real-evaluated 
                points. It's therefore highly suggested to not change this value in most
                applications.
            
            preprocessing (Literal["estimator_default"], optional):
                Preprocessing strategy to apply. 
                Must be "estimator_default". Can be ignored by users.

            tune_space (Literal["default"], optional):
                Tune space to use.
                Must be "default". Can be ignored by users.
        '''
        self.name=name
        self.n_members=n_members
        self.save_path=save_path
        self.seed=seed
        self.time_limit=time_limit
        self.log=log
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.n_threads=n_threads
        self.device=device
        self.meta_surrogate_model=meta_surrogate_model
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_seed=meta_seed
        self.preprocessing=preprocessing
        self.tune_space=tune_space



class StandardEnsembleInitializer:
    def __init__(
        self,
        save_path: str | Path,
        name: str = "random_ens",
        n_members: int = 16,
        preprocessing: PreprocessingStrategy = "estimator_default",
        tune_space: str = "default",
        seed: int = 0,
        time_limit: int = 10_000_000,
        log: int = 20,
        raise_error_fit_member: bool = False,
        raise_error_void_ensemble: bool = True,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto"
    ):
        '''
        Draw random hyperparameters configurations from the search space and ensemble them.
        
        Parameters:

            save_path (str | Path):
                Path of the folder where the ensemble members are saved.
                These will be serialized with the pickle module in files
                named after the members (name ensemble + number member).
                Note that the folder is created if not existent.
            
            name (str, optional): 
                Name of the ensemble that is used as root for members name.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of hps configuration to derive.

            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy applied to the data.
                It is recommended to disable preprocessing ("no") when the classifier is not early-stopped, 
                and reproducing and applying it beforehand when possible, for efficiency reasons. 
                In this case, preprocessing is repeated for every member of the ensemble.
                When early stopping is enabled, different validation splits are used. 
                To ensure preprocessing is applied correctly across these splits, this API should be used.
                Custom preprocessing can be applied by providing a fixed validation set and performing preprocessing beforehand. 
                However, this approach is suboptimal, as it does not exploit the variability introduced by different data splits.
                Currently, custom preprocessing is not supported when using different validation splits.

            tune_space (str, optional):
                Tune space to use. For GBDTs multiple pre-defined tune space are available.
                These are strings "c{integer}". Use "default" string to use the default space
                for every classifier. It's suggested to leave the default, since this parameter
                will be probably deprecated in future. 
            
            seed (int, optional):
                Seed controlling the randomness inherent to the hyperparameter 
                drawing process, the estimator and to the validation sets splits 
                when early stop is enabled.

            time_limit (int, optional):
                Time in seconds to spend for ensemble construction.
                The ensemble building process is stopped when this limit is violated.
                The default is 10 million equal to 115 days approximately, meaning no limit.

            log (int, optional):
                Level of the internal logger. The default assures logs at info level.
                To suppress it set a value of 40 or 50.

            raise_error_fit_member (bool, optional):
                Whether to ignore fit errors in the ensemble building process.
                If True the process is blocked when an fit error is encountered.
                Note that time limit errors are considered as fit errors here.
                If False the process goes on.

            raise_error_void_ensemble (bool, optional):
                Whether to raise an error when a void ensemble is obtained.
                This can be due to time limit or fit failings.

            n_threads (int, optional):
                Number of threads used to parallelize the classifiers fitting process.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device where to fit the model(s). 
                Note that for some estimators cannot be run on "cuda" raising an error.
                If "auto" then it selects cuda if available AND the estimator requires GPU else cpu.
        '''
        self.name=name
        self.n_members=n_members
        self.save_path=save_path
        self.preprocessing=preprocessing
        self.tune_space=tune_space
        self.seed=seed
        self.time_limit=time_limit
        self.log=log
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.n_threads=n_threads
        self.device=device