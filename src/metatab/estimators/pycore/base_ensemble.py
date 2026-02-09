from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.estimators.utils.general import learn_sklearn_features_attributes
from metatab.estimators.utils.general import check_predict_features
from metatab.estimators.core.configurations import EarlyStopConfiguration, EnsembleConfiguration
from metatab.estimators.params.utils import pick_estimator_tune_space
from metatab.estimators.utils.general import check_meta_tuning_options

if TYPE_CHECKING:
    import numpy as np
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.estimators.estimators import EnsembledEstimator
    from metatab.metatab_utils.types import XType, YType



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
        meta_strategy: MetaStrategy = "random_uniform_from_best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_surrogate_model: None | str | Path = None,
        meta_seed: int = 42,
        preprocessing: PreprocessingStrategy = "estimator_default",
        tune_space: Literal["default"] = "default"
    ):
        '''
        The meta ensembled estimators are backed by a meta-learning framework that suggests,
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
            We highly suggest to NOT preprocess the microbial profiles apart 
            expressing them in the "relative" format (i.e rows summing to 1). 
            This is because we automatically select the most appropiate preprocessing scheme for the classifier.
            In addition we learn the data metafeatures before preprocessing. 
            Therefore a custom pre-preprocessing can potentially hurt performance.


        Parameters:

            save_path (str | Path):
                Folder where ensemble members are saved. 
                Members are serialized with pickle and named using the ensemble name plus a member number. 
                The folder is created if it does not exist. 
            
            name (str, optional): 
                Name of the ensemble, used as a prefix for member filenames.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of hps configuration to derive.

            seed (int, optional):
                Seed controlling the randomness inherent to the classifier and validation splits when used.

            time_limit (int, optional):
                Time in seconds to spend for ensemble construction.
                The ensemble building process is stopped when this limit is violated.
                The default is 10 million equal to 115 days approximately, meaning no limit.

            log (int, optional):
                Logging level. 
                Default provides info-level logs. 
                Set to 40 or 50 to suppress logging.

            raise_error_fit_member (bool, optional):
                Whether to stop the process when a member fails to fit. 
                Time-limit errors are also considered fit errors. 
                If False, the process continues despite failures.

            raise_error_void_ensemble (bool, optional):
                Whether to raise an error when the fit process fails for all members.

            n_threads (int, optional):
                Number of threads used to parallelize the classifiers fitting process.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device to fit the model(s) on.
                - "cpu" or "cuda" explicitly selects the device.
                - "auto" uses GPU if available and supported by the classifier; otherwise CPU.

            meta_strategy (MetaStrategy, optional):
                Set the strategy used by the metalearning framework to select points.
                - "best": select the top-n configurations.
                - "random_from_best": random selection from the top.
                - "uniform_from_best": uniform step selection from the top.
                - "random_uniform_from_best": random selection within uniform intervals from the top.
                
            meta_strategy_params (None | MetaStrategyParams, optional):
                Meta strategy specifics in form of dataclass.
                If None the default specifics are applied.

            meta_surrogate_model (None | str | Path, optional):
                Surrogate model to use for point performance prediction.
                If str or Path, then the object pointed by the path is used as surrogate model.
                This must be a joblib serialized object.
                If None the "default" surrogate model is used.

            meta_seed (int, optional):
                Seed specifically and only used to draw candidate points.
                The default value of 42 is the one used to generate our prior.
                Using this value allows to evaluate points tested beforehand. 
                Therefore is highly suggested to not modify this value.
                Note: If the parameter is left at its default but the number of 
                candidate points (set via `meta_strategy_params`) differs from the 
                default of 1500, the following occurs.
                - If the number of candidate points is less than 1500, 
                a subset of the prior points is selected.
                - If the number exceeds 1500, "new" points are drawn in addition 
                to the prior points.
            
            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy to apply. 
                Is highly suggested to leave "estimator_default",
                since other options could affect performance negatively.
                See the user note above for details.

            tune_space (Literal["default"], optional):
                Must be "default". Can be ignored by users.


        ## Attributes:

            is_void_ (bool): 
                Flag informing whether the ensemble is void.

            is_cleaned_ (bool): 
                Flag informing whether the ensemble models have been 
                deleted from disk using the "delete_models_from_disk" method.
            
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
            
            n_features_in_ (int): 
                Number of features seen at fit level.
            
            feature_names_in_ (np.ndarray): 
                Names of features seen during fit.
                Exists only when fitted on a pandas DataFrame with strings as columns.
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
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_surrogate_model=meta_surrogate_model
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
                Folder where ensemble members are saved. 
                Members are serialized with pickle and named using the ensemble name plus a member number. 
                The folder is created if it does not exist.
            
            name (str, optional): 
                Name of the ensemble, used as a prefix for member filenames.

            n_members (int, optional):
                Number of ensemble members.
                In other terms the number of hps configuration to use.

            preprocessing (PreprocessingStrategy, optional):
                Preprocessing strategy applied to the data.
                - For non early stopped classifiers, it is recommended to disable preprocessing ("no") 
                and apply it beforehand, when possible, for efficiency, as otherwise it is 
                repeated for every ensemble member.
                - When early stopping is enabled, different validation splits are used.
                Using this API ensures preprocessing is applied correctly across these splits.
                - Custom preprocessing cannot be specified via this.

            tune_space (str, optional):
                Pre-defined tuning space to use. 
                Use strings like "c{integer}" to select a specific space. 
                For non-GBDT classifiers, only "default" or "c0" can be used. 
                The default is recommended, as this parameter may be deprecated in the future 
                and internal tests showed no major performance differences among alternative spaces.
            
            seed (int, optional):
                Seed controlling the randomness inherent to the estimators, hyperparameter 
                drawing process, and validation splits for early stopped estimators.

            time_limit (int, optional):
                Time in seconds to spend for ensemble construction.
                The ensemble building process is stopped when this limit is violated.
                The default is 10 million equal to 115 days approximately, meaning no limit.

            log (int, optional):
                Logging level. 
                Default provides info-level logs. 
                Set to 40 or 50 to suppress logging.

            raise_error_fit_member (bool, optional):
                Whether to stop the process when a member fails to fit. 
                Time-limit errors are also considered fit errors. 
                If False, the process continues despite failures.

            raise_error_void_ensemble (bool, optional):
                Whether to raise an error when the fit process fails for all members.

            n_threads (int, optional):
                Number of threads used to parallelize the classifiers fitting process.

            device (Literal["cpu", "cuda", "auto"], optional):
                Device where to fit the model(s). 
                Note that for some estimators cannot be run on "cuda" raising an error.
                If "auto" then it selects cuda if available AND the estimator requires GPU else cpu.

                
         ## Attributes:

            is_void_ (bool): 
                Flag informing whether the ensemble is void.
            
            is_cleaned_ (bool): 
                Flag informing whether the ensemble models have been 
                deleted from disk using the "delete_models_from_disk" method.
            
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
            
            n_features_in_ (int): 
                Number of features seen at fit level.
            
            feature_names_in_ (np.ndarray): 
                Names of features seen during fit.
                Exists only when fitted on a pandas DataFrame with strings as columns.
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



class BaseEnsemble(ClassifierMixin, BaseEstimator):
    def fit(
        self,
        X: XType,
        y: YType,
        early_stop_configuration: None | EarlyStopConfiguration = None
    ):
        check_X_y(X, y, dtype=None, ensure_all_finite=False)

        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity
        
        if self.type_ensemble == "random":
            meta_ensemble_parameters = {}
        else:
            # only this check is necessary here
            check_meta_tuning_options(
                estimator=self.type_estimator,
                preprocessing=self.preprocessing,
                tune_space=self.tune_space
            )

            meta_ensemble_parameters = {
                "meta_strategy": self.meta_strategy,
                "meta_strategy_params": self.meta_strategy_params,
                "meta_surrogate_model": self.meta_surrogate_model,
                "meta_seed": self.meta_seed
            }

        ens_conf = EnsembleConfiguration(
            name=self.name,
            algo=self.type_ensemble,
            n_members=self.n_members,
            save_path=self.save_path,
            params_distributions=pick_estimator_tune_space(self.type_estimator, self.tune_space),
            time_limit=self.time_limit,
            log=self.log,
            raise_error_fit_member=self.raise_error_fit_member,
            raise_error_void_ensemble=self.raise_error_void_ensemble,
            **meta_ensemble_parameters
        )

        estimator: EnsembledEstimator = self.myclass(
            preprocessing=self.preprocessing,
            seed=self.seed,
            n_threads=self.n_threads,
            device=self.device,
            early_stop_configuration=early_stop_configuration,
            ensemble_configuration=ens_conf
        )

        self.estimator_ = estimator.fit(X, y)
        
        self.classes_ = le.classes_
        sklearn_info = learn_sklearn_features_attributes(X)
        ensemble_info = self.estimator_.collect_ensemble_fit_info()
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
        '''Delete the ensemble models from disk'''        
        self.estimator_.estimator_.delete_models_from_disk()
        self.is_cleaned_ = True