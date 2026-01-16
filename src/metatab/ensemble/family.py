from __future__ import annotations

import sys
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y
from metatab.metatab_utils.general import subset_xy, subset_2d
from metatab.metalearning.metafeatures import CustomMFE
from metatab.estimators.params.utils import pick_estimator_tune_space
from metatab.estimators.core.configurations import EnsembleConfiguration, EarlyStopConfiguration
from metatab.estimators.utils.pick import pick_estimator_class

from metatab.estimators.utils.general import (
    check_predict_features,
    learn_sklearn_features_attributes
)

from metatab.ensemble.utils import BagCV
from metatab.ensemble.configuration import UserEnsembleConfiguration, CollectionUserEnsembleConfiguration

if TYPE_CHECKING:
        from metatab.estimators.estimators import EnsembledEstimator
        from metatab.metatab_utils.types import XType, YType



class FamilyEnsembleEstimator:
    '''
    Hierarchical ensemble of inner or first level ensembles.

    Parameters:
        name (str): Family ensemble name.
        
        configuration (UserEnsembleConfiguration | CollectionUserEnsembleConfiguration): 
            Configuration instace/s describing and characterinzing the first 
            level ensemble/s that form the ensemble family.
        
        save_path (str | Path):
            Path to the directory where ensemble members will be saved. 
            For each first-level ensemble in the family, 
            a subfolder with the ensemble name is created. 
            Inside each subfolder, the first-level ensemble members are 
            serialized using pickle and stored in files named after the member 
            (ensemble name + member ID).
        
        bag_cv (None | BagCV, optional):
            A dataclass specifying the cross-validation procedure used to subset the training rows. 
            The purpose is to create diverse subsets of the data using CV; 
            this diversity can help decorrelate the predictions of the first-level ensembles.
            If None, no subsampling is performed and all first-level ensembles are trained on the same rows.

        feature_space_ratio (float, optional):
            A value in the interval (0, 1] indicating the fraction 
            of the feature space to randomly select for each first-level ensemble. 
            This randomization strategy helps increase diversity and reduce 
            correlation among first-level ensemble predictions.
            If 1, no subsetting is performed and all first-level ensembles 
            are trained on the full feature space.

        raise_error_void_ensemble (bool, optional):
            Whether to raise an error when a void ensemble is obtained.
            This can be due to time limit or fit failings.

        seed (int, optional):
            Seed used to control the feature space randomization.

        time_limit (int, optional):
            Time in seconds to spend for ensemble construction.
            It's important to know that this value is equally divided among the first-level ensembles. 
            This choice aims to favor scenarios with more partial ensembles than few complete ones. 
            These ensemble families have more diversification and should give better performance.
            The ensemble building process is stopped when this limit is violated.
            The default is 10 million equal to 115 days approximately, meaning no limit.

        n_jobs (int, optional):
            Number of threads used to parallelize the internal classifiers.

        log (int, optional):
            Level of the internal logger.
            Controls both the family and first-level ensembles logging.
            The default assures logs at info level.
            To suppress it set a value of 40 or 50.

    ## Attributes:

        ensembles_ (list[EnsembleEstimator]):
            List of the internal esembles.
        
        is_void_ (bool): 
            Flag informing about the emptyness of the ensemble.

        is_cleaned_ (bool):
            Flag informing whether the ensemble models on disk have been deleted
            using the "delete_models_from_disk" method.

        fit_time_ (float): 
            Ensemble fit time. 
            It is derived as the sum of the internal ensemble fit times.
        
        successful_members_ (dict[str, list[str]]):
            Dict of first-level ensemble names as keys and list of the successfully fitted member names as value.
        
        failed_members_ (dict[str, list[str]]): 
            Dict of first-level ensemble names as keys and list of failed member names as value.
        
        successful_hps_confs_ (dict[str, list[dict]]): 
            Dict of first-level ensemble names as keys and list of successful hps points as value.
        
        failed_hps_confs_ (dict[str, list[dict]]): 
            Dict of first-level ensemble names as keys and list of failed hps points as value. 
        
        df_members_ (pd.DataFrame): 
            DataFrame with info about all members fit process.

        classes_ (np.ndarray): The array of class labels learnt at fit time.
        
        n_features_in_ (int): Number of features seen at fit level.
       
        feature_names_in_ (np.ndarray): 
            Names of the features seen at fit level.
            This attribute exists only when the instance is fitted with pandas dataframe with all string columns.
    '''
    def __init__(
        self,
        name: str,
        configuration: UserEnsembleConfiguration | CollectionUserEnsembleConfiguration,
        save_path: str | Path,
        bag_cv: None | BagCV = None,
        feature_space_ratio: float = 1,
        raise_error_void_ensemble: bool = True,
        seed: int = 0,
        time_limit: int = 10_000_000,
        n_jobs: int = 1, 
        log: int = 20
    ):
        self.name=name
        self.configuration=configuration
        self.save_path=save_path
        self.bag_cv=bag_cv
        self.feature_space_ratio=feature_space_ratio
        self.raise_error_void_ensemble=raise_error_void_ensemble
        self.seed=seed
        self.time_limit=time_limit
        self.n_jobs=n_jobs
        self.log=log


    def fit(self, X: XType, y: YType) -> "FamilyEnsembleEstimator":
        '''
        Fit the ensemble

        Parameters:
            X (XType): Train data.
            y (YType): Train labels.

        Returns:
            self
        '''
        self._check_initialization_inputs()
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        
        # encode y
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.Series(y) if isinstance(X, pd.DataFrame) else y  # for Xy "type" uniformity

        confs = [self.configuration]\
            if isinstance(self.configuration, UserEnsembleConfiguration)\
            else self.configuration.configurations

        n_confs = len(confs)
        logger = self._get_logger()
        self._save_path = Path(self.save_path) if isinstance(self.save_path, str) else self.save_path

        logger.info("Starting Preparation phase.")

        # get rows indexes
        if self.bag_cv:
            if (bag_n_iter := self.bag_cv.n_repeats * self.bag_cv.n_folds) < n_confs:
                raise ValueError(
                    f"BagCV total iterations ({bag_n_iter}) < number of input configurations ({n_confs})."
                )
            
            bag_cv_splitter = RepeatedStratifiedKFold(
                n_repeats=self.bag_cv.n_repeats,
                n_splits=self.bag_cv.n_folds,
                random_state=self.bag_cv.seed
            )

            row_idx = [t[0] for i, t in enumerate(bag_cv_splitter.split(X, y)) if i < n_confs]
        else:
            row_idx = [None] * n_confs

        logger.info("Preparation phase: obtained train rows indices.")

        # get cols indexes
        if self.feature_space_ratio != 1:
            rng = np.random.default_rng(self.seed)
            n_features = X.shape[1]
            n_subset_features = int(n_features * self.feature_space_ratio)
            col_idx = [
                rng.choice(n_features, size=n_subset_features, replace=False)
                for _ in range(n_confs)
            ]
        else:
            col_idx = [None] * n_confs
        
        # we need the cols indices for inference
        self._col_idx = col_idx
        logger.info("Preparation phase: obtained train cols indices.")

        # get metafeatures
        if self.bag_cv is None and self.feature_space_ratio == 1:
            mfe = CustomMFE()
            metafeatures, _ = mfe.fit(X, y).extract()
        else:
            metafeatures = None

        logger.info("Preparation phase: processed metafeatures directive.")

        # TODO: for now we do not optimize-cache the candidate points drawing process
        # since it's more difficult to control this and the raw process is quite fast        
        meta_candidate_points = None
        logger.info("Preparation phase: processed meta_candidate_points directive.")

        # we give equal time to favor more ensembles with less members 
        # than fewer ensembles with all members
        time_ens = int(self.time_limit / n_confs)

        ens_classifiers = [
            self._get_ensemble_classifier(conf, metafeatures, meta_candidate_points, time_ens) 
            for conf in confs
        ]
        
        assert len(set([len(ens_classifiers), len(row_idx), len(col_idx)])) == 1
        
        logger.info("Completed Preparation phase.\n")
        logger.info("Starting ensemble building process.\n")

        self.ensembles_: list[EnsembledEstimator] = []
        for rows, cols, ens_classifier in zip(row_idx, col_idx, ens_classifiers):
            X_clf, y_clf = subset_xy(X, y, rows, cols)
            self.ensembles_.append(ens_classifier.fit(X_clf, y_clf))

        self._collect_inner_ensembles_info()
        self.is_cleaned_ = False        
        for k, v in learn_sklearn_features_attributes(X).items():
            setattr(self, k, v)
        self.classes_ = le.classes_

        logger.info(f"Completed family ensemble building process in {round(self.fit_time_ / 60, 2)} minutes.")
        
        if self.raise_error_void_ensemble and self.is_void_:
            raise ValueError("All inner ensembles are void.")
        
        return self


    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict class labels for X.
        The labels are inferred from the ensemble average probabilities.

        Parameters:
            X (XType): Input samples.
            do_check (bool, optional): 
                Whether to check for fitted and void esemble before prediction.

        Returns:
            np.ndarray: The predicted classes.
        '''
        self._check_on_predict_calls(X)
        return np.argmax(self.predict_proba(X), axis=1)


    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            np.ndarray: The predicted probabilities.
        '''
        self._check_on_predict_calls(X)
        return np.stack(
            [
                ens.predict_proba(subset_2d(X, idx_rows=None, idx_cols=cols)) 
                for ens, cols in zip(self.ensembles_, self._col_idx) 
                if not ens.estimator_.is_void_
            ],
            axis=0
        ).mean(axis=0)


    def get_inner_ensembles_predicted_probabilities(self, X: XType) -> dict[str, np.ndarray]:
        '''
        Get first-level ensemble class probabilities for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            dict[str, np.ndarray]: 
            A dict with the first-level ensemble names as keys and predictions as values.
        '''
        self._check_on_predict_calls(X)
        return {
            ens.estimator_.name: ens.predict_proba(subset_2d(X, idx_rows=None, idx_cols=cols)) 
            for ens, cols in zip(self.ensembles_, self._col_idx) 
            if not ens.estimator_.is_void_
        }


    def get_members_predicted_probabilities(self, X: XType) -> dict[str, dict[str, np.ndarray]]:
        '''
        Get first-level ensemble members class probabilities for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            dict[str, dict[str, np.ndarray]]: 
            A dict with the first-level ensemble names as keys 
            and dicts member_name:probabilities as values.
        '''
        self._check_on_predict_calls(X)
        return {
            ens.estimator_.name: ens.get_members_predicted_probabilities(subset_2d(X, idx_rows=None, idx_cols=cols)) 
            for ens, cols in zip(self.ensembles_, self._col_idx)
            if not ens.estimator_.is_void_
        }
    

    def delete_models_from_disk(self) -> None:
        '''Deletes the ensemble models from disk'''
        check_is_fitted(self, "ensembles_")
        
        if self.is_cleaned_:
            return None
        
        if self.is_void_:
            self.is_cleaned_ = True
            warnings.warn("The ensemble is void. No model to delete.")
            return None

        for ens in self.ensembles_:
            ens.estimator_.delete_models_from_disk()
        
        self.is_cleaned_ = True
    

    def save(self, filepath: str | Path, check_is_fitted: bool = False) -> None:
        '''
        Serialize the instance using pickle.
        Parameters:
            filepath (str | Path): 
                File in which to serialize the instance.
            check_is_fitted (bool, optional):
                Whether to check if the instance is fitted prior serialization.
        '''
        if check_is_fitted and not hasattr(self, "ensembles_"):
            raise ValueError("The ensemble instance is not fitted.")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        
    
    def _check_on_predict_calls(self, X: XType) -> None:
        check_is_fitted(self, "ensembles_")
        self._check_void_ensemble()
        self._check_cleaned_ensemble()
        check_predict_features(self, X)


    def _check_void_ensemble(self) -> None:
        if self.is_void_:
            raise ValueError("The ensemble is void.")
        

    def _check_cleaned_ensemble(self) -> None:
        if self.is_cleaned_:
            raise ValueError("The ensemble models have been deleted.")


    def _collect_inner_ensembles_info(self) -> None:
        '''Collect and set in self all the inner esembles info'''
        successful_members = {}
        failed_members = {}
        successful_hps_confs = {}
        failed_hps_confs = {}
        df_members_list = []

        fit_time_total = 0
        void_flags = []

        for ens in self.ensembles_:
            estimator = ens.estimator_
            name = estimator.name
            void_flags.append(estimator.is_void_)
            fit_time_total += estimator.fit_time_
            successful_members[name] = estimator.successful_members_
            failed_members[name] = estimator.failed_members_
            successful_hps_confs[name] = estimator.successful_hps_confs_
            failed_hps_confs[name] = estimator.failed_hps_confs_
            df_members_list.append(estimator.df_members_.assign(ensemble=name))

        self.is_void_ = all(void_flags)
        self.fit_time_ = fit_time_total
        self.successful_members_ = successful_members
        self.failed_members_ = failed_members
        self.successful_hps_confs_ = successful_hps_confs
        self.failed_hps_confs_ = failed_hps_confs
        self.df_members_ = pd.concat(df_members_list, axis=0, ignore_index=True)


    def _get_ensemble_classifier(
        self, 
        conf: UserEnsembleConfiguration,
        metafeatures: None | dict,
        meta_candidate_points: None | list[dict],
        time_ens: int
    ) -> EnsembledEstimator:
        '''
        Method that process the user ensemble estimator configuration, the family instance
        attributes, metafeatures and meta candidate points to initialize and get the 
        concrete "MyEnsembled*" estimator instances.
        '''
        estimator_class = pick_estimator_class(conf.estimator, "ensemble")

        ens_conf = EnsembleConfiguration(
            name=conf.name,
            algo=conf.algo,
            n_members=conf.n_members,
            save_path=self._save_path / conf.name,
            params_distributions=pick_estimator_tune_space(conf.estimator, conf.tune_space),
            meta_strategy=conf.meta_strategy,
            meta_strategy_params=conf.meta_strategy_params,
            meta_surrogate_model=conf.meta_surrogate_model,
            meta_seed=conf.meta_seed,
            meta_features=metafeatures,
            meta_candidate_points=meta_candidate_points,
            time_limit=time_ens,
            log=self.log,
            raise_error_fit_member=False,
            raise_error_void_ensemble=False
        )

        if conf.early_stop_on_validation_set:
            esc = EarlyStopConfiguration(
                early_stop_rounds=conf.early_stop_rounds,
                validation_set_size=conf.validation_set_size
            )
        else:
            esc = None

        return estimator_class(
            preprocessing=conf.preprocessing, 
            seed=conf.seed,
            n_threads=self.n_jobs,
            device=conf.device,
            early_stop_configuration=esc,
            ensemble_configuration=ens_conf
        )


    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.log)
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    
    def _check_initialization_inputs(self) -> None:
        if not isinstance(
            self.configuration, 
            (UserEnsembleConfiguration, CollectionUserEnsembleConfiguration)
        ):
            raise ValueError(
                "'configuration' parameter must be a UserEnsembleConfiguration" +
                " or CollectionUserEnsembleConfiguration instance."
            )        
        if not 0 < self.feature_space_ratio <= 1:
            raise ValueError("'feature_space_ratio' must be in (0, 1].")