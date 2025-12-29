from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.estimators.utils.general import check_predict_features, check_y_is_integer_encoded
from metatab.estimators.core.configurations import EarlyStopConfiguration, EnsembleConfiguration

if TYPE_CHECKING:
    import numpy as np
    from pathlib import Path
    from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.estimators.estimators import EnsembledEstimator
    from metatab.metatab_utils.types import XType, YType



class MetaEnsembleBaseEstimator(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        save_path: str | Path,
        name: str = "meta_ens",
        n_members: int = 16,
        meta_surrogate_model: None | str | Path = None,
        meta_strategy: MetaStrategy = "random_uniform_from_best",
        meta_strategy_params: None | MetaStrategyParams = None,
        meta_seed: int = 42,
        seed: int = 0,
        time_limit: int = 10_000_000,
        log: int = 20,
        raise_error_fit_member: bool = False,
        raise_error_void_ensemble: bool = True,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto"
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

            meta_surrogate_model (None | str | Path, optional):
                Surrogate model to use for point performance prediction.
                If str or Path, then the object pointed by the path is used as surrogate model.
                This must be a joblib serialized object.
                If None the "default" surrogate model is used.

            meta_strategy (MetaStrategy, optional):
                Set the strategy used by the metalearning framework to select points.
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


        ## Attributes:

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
        self.name=name
        self.n_members=n_members
        self.save_path=save_path
        self.meta_surrogate_model=meta_surrogate_model
        self.meta_strategy=meta_strategy
        self.meta_strategy_params=meta_strategy_params
        self.meta_seed=meta_seed
        self.seed=seed
        self.time_limit=time_limit
        self.log=log
        self.raise_error_fit_member=raise_error_fit_member
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.n_threads=n_threads
        self.device=device


    def fit(
        self,
        X: XType, 
        y: YType,
        preprocessing: PreprocessingStrategy,
        concrete_estimator_cls: EnsembledEstimator,
        tuning_params: dict,
        early_stop_configuration: None | EarlyStopConfiguration
    ):
        '''
        Fit the metaensemble estimator.

        Parameters:
            X (XType): Train data.
            
            y (YType): Train labels.
            
            preprocessing (PreprocessingStrategy):
                Preprocessing strategy to use.

            concrete_estimator_class (Estimator): 
                Ensembled estimator class to instantiate.
            
            tuning_params (dict): 
                Tune space. Must be compatible with the surrogate model.
            
            early_stop_configuration (None | EarlyStopConfiguration):
                Must be implemented by the early stopped concrete estimators.
        
        Returns:
            self
        '''
        check_y_is_integer_encoded(y)
        check_X_y(X, y, dtype=None, ensure_all_finite=False)

        ens_conf = EnsembleConfiguration(
            name=self.name,
            algo="meta",
            n_members=self.n_members,
            save_path=self.save_path,
            params_distributions=tuning_params,
            meta_strategy=self.meta_strategy,
            meta_strategy_params=self.meta_strategy_params,
            meta_surrogate_model=self.meta_surrogate_model,
            meta_seed=self.meta_seed,
            time_limit=self.time_limit,
            log=self.log,
            raise_error_fit_member=self.raise_error_fit_member,
            raise_error_void_ensemble=self.raise_error_void_ensemble
        )

        estimator: EnsembledEstimator = concrete_estimator_cls(
            preprocessing=preprocessing,
            seed=self.seed,
            n_threads=self.n_threads,
            device=self.device,
            early_stop_configuration=early_stop_configuration,
            ensemble_configuration=ens_conf
        )

        self.estimator_ = estimator.fit(X, y)
        sklearn_info = self.estimator_.collect_sklearn_fit_info()
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
        '''Delete the ensemble models from disk.'''
        self.estimator_.estimator_.delete_models_from_disk()
        self.is_cleaned_ = True