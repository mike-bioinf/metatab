from __future__ import annotations

import numpy as np
import pandas as pd
from math import inf
from copy import deepcopy
from typing import Generator, TYPE_CHECKING, Literal
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from estimators.estimators.utils import get_fresh_random_state

from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    precision_recall_curve
)

from scipy.stats._distn_infrastructure import (
    rv_continuous_frozen, 
    rv_discrete_frozen
)

if TYPE_CHECKING:
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import Pipeline
    from estimators.estimators.types import BoostedClassifier



class MyRandomSearchCV:

    def __init__(
        self,
        classifier: BoostedClassifier,
        fixed_params_classifier: dict,
        param_distributions: dict,
        splitter: RepeatedStratifiedKFold,
        preprocessing_pipeline: None | Pipeline,
        scorer: Literal["auc", "logloss", "aucpr"],
        n_iter: int,
        refit: bool,
        seed: int
    ):
        '''
        Class that executes random search cv allowing and enforcing (is mandatory)
        early stop for boosted tree models.
        In detail the class implements a 3-set strategy where the data is split
        initially in 2 sets during cv and then the train section is further split
        in real train and early-stop validation set.

        Note: the splitter must receive in input a seed and NOT a RandomState.
        In the latter case the search is not uniform accross CVs.


        Parameters
        ---------------
        classifier (BoostedClassifier):
            Class used to build the classifier (XGBClassifier or)

        fixed_params_classifier (dict):
            Dict of the fixed classifier parameters, aka the parameters
            that must not be tuned.
        
        param_distributions (dict):
            Dict of the distributions of the parameters to tune.
            Must follow the RandomizedSearchCV related sklearn API.
        
        splitter (RepeatedStratifiedKFold):
            Instance of "RepeatedStratifiedKFold". Importantly must
            be created using a seed (int) as random_state.
            If this is not true then the cvs for the different HPs
            are not the same in terms of data splitting.
        
        preprocessing_pipeline (None | Pipeline):
            Sklearn pipelines used to preprocess the features sets created
            during the search. If None no preprocessing is applied.
        
        scorer (Literal["auc", "logloss", "aucpr"]):
            Metric used to validate the model on the validation fold.
            Currently "aucpr" is not implemented.
        
        n_iter (int): 
            Number of random search iterations.
        
        refit (bool):
            Whether to refit the classifier using the best HPs 
            after the search process.
        
        seed (int):
            Integer for reproducibility. 
            Affects the data splitting and HP space sampling procedures.

            
        Attributes
        ------------
        best_score_ (float): Best validation score from the search.
        
        best_params_ (dict): Best combination of HPs from the search.
        
        best_estimator_ (BoostedClassifier): 
            Refitted classifier with the best HPs.
            Present only if refit is True.
        
        preprocessing_pipeline_ (Pipeline | None): 
            Refitted preprocessing pipeline over the final training data.
            Present only when refit is True.
        
        classes_ (np.ndarray): Array with the sorted classes.

        n_classes_ (int): Number of classes.
        '''
        self.classifier = classifier
        self.fixed_params_classifier = fixed_params_classifier
        self.param_distributions = param_distributions
        self.splitter = splitter
        self.preprocessing_pipeline = preprocessing_pipeline
        self.scorer = scorer
        self.n_iter = n_iter
        self.refit = refit
        self.seed = seed
        self._check_preprocessing_pipeline()



    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyRandomSearchCV":
        '''
        Execute the random search sampling "n_iter" combinations.
        Set the best combinations of HPs along with other info in attrs.
        '''
        self.classes_ = y.unique()
        self.n_classes_ = self.classes_.size
        rng_sampling = np.random.default_rng(self.seed)

        best_score = -inf
        best_hps = None

        for sample_tune_params in self._sample_from_distributions(self.n_iter, rng_sampling):
            # we use the same fresh RandomState for every cv classifier,
            # this maximize the internal cv entropy,
            # while preserving uniformity accross cvs.
            fixed_params = deepcopy(self.fixed_params_classifier)
            random_state_cv = get_fresh_random_state(fixed_params["random_state"])
            fixed_params["random_state"] = random_state_cv

            # build the classifier with fixed plus tuning params
            clf: BoostedClassifier = self.classifier(**fixed_params, **sample_tune_params)
            cv_scores = []

            # perform cv with a single HP combination
            for train_idx, test_idx in self.splitter.split(X, y):
                X_train_fold, y_train_fold = X.iloc[train_idx, :], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
                
                # getting validation set for early stopping
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_fold,
                    y_train_fold,
                    train_size=0.75,
                    random_state=int(random_state_cv.randint(0, 2**32, dtype=np.int64)),
                    stratify=y_train_fold
                )
                
                # apply preprocessing
                X_train_trans, X_val_trans, X_test_trans = self._preprocess(
                    X_train, 
                    X_val, 
                    X_test
                )

                # TODO: check that the fit API is respected for every boosted model used
                # verbose if for xgboost and if not set here is not possible to avoid logs
                clf.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)], verbose=False)
                pred_proba = clf.predict_proba(X_test_trans)
                cv_scores.append(self.compute_score(pred_proba, y_test, self.n_classes_))
            
            # check if the HPs combination is the best among the tested ones 
            # in case of equality we take the current best one (is unlikely anyway)
            sample_score = sum(cv_scores) / len(cv_scores)
        
            if sample_score > best_score:
                best_score = sample_score
                best_hps = sample_tune_params
        
        # set the best attrs
        self.best_score_ = best_score
        self.best_params_: dict = best_hps

        # refit with best hps
        if self.refit:
            # take a new final validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                train_size=0.75,
                random_state=self.seed,
                stratify=y
            )

            X_train_trans, X_val_trans = self._preprocess(X_train, X_val)
            # we crate a new reference for the refitted preprocessing pipeline 
            # to remark that is learned and must be used on test data
            self.preprocessing_pipeline_ = self.preprocessing_pipeline
            best_estimator: BoostedClassifier = self.classifier(**self.fixed_params_classifier, **best_hps)
            best_estimator.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)], verbose=False)
            self.best_estimator_ = best_estimator
        
        return self



    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Predict class probabilities for X'''
        check_is_fitted(self, "best_estimator_")
        if self.preprocessing_pipeline_ is not None:
            X = self.preprocessing_pipeline_.transform(X)
        return self.best_estimator_.predict_proba(X, **kwargs)



    def _preprocess(
        self,
        train: pd.DataFrame,
        *others: tuple[pd.DataFrame]
     ) -> list[np.ndarray]:
        '''
        Fit and apply the preprocessing pipeline on X train and transform the other X sets.
        If the preprocessing pipeline is None, the sets are casted to numpy arrays and returned.
        In every case the train data is returned first and the others in input order.
        '''
        preprocessed_data = []
        if self.preprocessing_pipeline is None:
            preprocessed_data.append(train)
            preprocessed_data.extend(others)
            preprocessed_data = self._to_numpy(*preprocessed_data)
        else:
            train_trans = self.preprocessing_pipeline.fit_transform(train)
            preprocessed_data = [self.preprocessing_pipeline.transform(x) for x in others]
            preprocessed_data.insert(0, train_trans) 
        return preprocessed_data



    def compute_score(self, y_pred: np.ndarray, y_true: pd.Series, n_classes: int) -> float:
        if self.scorer == "auc":
            return roc_auc_score(
                y_true=y_true,
                y_score=y_pred[:, 1] if n_classes == 2 else y_pred,
                average=None if n_classes == 2 else "macro",
                multi_class="raise" if n_classes == 2 else "ovr"
            )
        elif self.scorer == "logloss":
            # we take the negative in order to maximize it
            return -1 * log_loss(y_true=y_true, y_pred=y_pred)
        elif self.scorer == "aucpr":
            raise NotImplementedError("aucpr scorer is yet to be implemented.")
        else:
            raise ValueError("Unsupported scorer.")
        


    def _sample_from_distributions(
        self, 
        n_samples: int, 
        rng: np.random.Generator
    ) -> Generator[dict, None, None]:
        '''
        Utility to sample from a dict of distibutions like 
        the ones accepted by RandomizedSearchCV.

        Parameters:
            dists (dict): RandomizedSearchCV-like distributions dict.
            n_samples (int): Number of returned samples.
            rng (Generator): numpy rng. 

        Returns:
            A generator that yield sampled valued in a dict.
        '''
        n_sampled = 0
        while n_sampled < n_samples:
            sample = self._sample_distribution(rng)
            n_sampled += 1
            yield sample 



    def _sample_distribution(self, rng: np.random.Generator) -> dict:
        '''
        Utility to sample from the distribution dict.
        Returns the sample as a dict of param:sampled_value.
        '''
        sample = {}
        for k, v in self.param_distributions.items():
            if isinstance(v, list) and len(v) == 1:
                sample[k] = v[0]
            elif isinstance(v, list):
                sample[k] = rng.choice(v)
            elif isinstance(v, (rv_continuous_frozen, rv_discrete_frozen)):
                sample[k] = self._to_number(v.rvs(random_state=rng))
            else:
                raise ValueError(
                    "Unsupported distribution object used in param_distributions parameter."
                )    
        return sample


    @staticmethod
    def _to_number(input: np.ndarray | float | int) -> float | int:
        '''Convert the potential numpy scalar/array output of rvs methods to float/int'''
        if isinstance(input, np.ndarray):
            if input.size != 1:
                raise ValueError("Expected a scalar array, instead size is greater than 1.")
            return input[0]
        elif hasattr(input, "dtype") and np.issubdtype(input.dtype, np.floating):
            return float(input)
        elif hasattr(input, "dtype") and np.issubdtype(input.dtype, np.integer):
            return int(input)
        else:
            return input
            

    @staticmethod
    def _to_numpy(*args) -> list:
        '''Convert pandas dataframe and series objects in numpy arrays'''
        casted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                casted_args.append(arg)
            elif isinstance(arg, (pd.Series, pd.DataFrame)):
                casted_args.append(arg.to_numpy())
            else:
                raise ValueError("Unsupported types in input.")
        return casted_args


    def _check_preprocessing_pipeline(self) -> None:
        '''Checks that the last step of the preprocessing pipeline is a transformer'''
        if self.preprocessing_pipeline is None:
            return None
        if not hasattr(self.preprocessing_pipeline[-1], "transform"):
            raise ValueError("The preprocessing pipeline does not have a trasformer in the last step.")