import numpy as np
import pandas as pd
from typing import Literal
from copy import deepcopy
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    RandomizedSearchCV, 
    train_test_split
)

from estimators.estimators.random_search import MyRandomSearchCV
from estimators.estimators.abstract_estimator import AbstractEstimator
from estimators.estimators.utils import add_string_to_params, create_default_pipeline

from estimators.estimators.params import (
    RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS, 
    ES_XGBCLASSIFIER_FIXED_PARAMS,
    XGBCLASSIFIER_FIXED_PARAMS,
    SKLEARN_RANDOM_SEARCH_FIXED_PARAMS,
    N_ITERATIONS_RANDOM_SEARCH
)



class MyRandomizedESXGBClassifier(AbstractEstimator):
    '''
    Class that uses a custom implementation of random search cv (MyRandomSearchCV)
    with the XGBClassifier. This custom implementation allows and enforces the 
    use of early stopping on a validation set for the ensemble building process.

    Attributes
    ------------------    
    estimator_ (MyRandomSearchCV): Fitted MyRandomSearchCV instance. 
    '''
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions: dict = RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = ES_XGBCLASSIFIER_FIXED_PARAMS
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomizedESXGBClassifier":
        fixed_params = _adjust_xgb_objective(self.fixed_params, y)
        fixed_params = _adjust_logloss_es_metric(fixed_params, y)

        # pass a seed here in splitter and NOT a RandomState
        splitter = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=self.seed)
        preprocessing_pipeline = self._create_preprocessing_pipeline()

        estimator = MyRandomSearchCV(
            classifier=XGBClassifier,
            fixed_params_classifier=fixed_params,
            param_distributions=self.params_distributions,
            preprocessing_pipeline=preprocessing_pipeline,
            splitter = splitter,
            scorer="logloss",
            n_iter=N_ITERATIONS_RANDOM_SEARCH,
            refit=True,
            seed=self.seed
        )

        self.estimator_ = estimator.fit(X, y)
        return self

    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        return super()._classic_predict_proba(X)
    
    def save(self, filepath: str | Path) -> None:
        super().save(filepath)       

    def _create_preprocessing_pipeline(self) -> Pipeline:
        return create_default_pipeline(self.preprocessing, "oversample")

    def _get_fitted_preprocessing_pipeline_or_estimator(self):
        return self.estimator_.preprocessing_pipeline_



class MyRandomizedXGBClassifier(AbstractEstimator):
    '''
    Class that implements the xgboost classifier in random search
    to tune the tunable HPs without early stopping.
    
    Attributes
    ------------------    
    estimator_ (RandomizedSearchCV): 
        Fitted RandomizedSearchCV on a Pipeline with XGBClassifier as last step.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions: dict = RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS,
        fixed_params: dict = XGBCLASSIFIER_FIXED_PARAMS  
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomizedXGBClassifier":
        fixed_params = _adjust_xgb_objective(self.fixed_params, y)
        
        estimator = RandomizedSearchCV(
            estimator=self._create_estimator(fixed_params),
            param_distributions=add_string_to_params(self.params_distributions, "xgbclassifier__"),
            cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=self.seed),
            **SKLEARN_RANDOM_SEARCH_FIXED_PARAMS,
        )

        self.estimator_ = estimator.fit(X, y)
        return self
    
    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return super()._classic_predict_proba(X)
    
    def save(self, filepath: str | Path) -> None:
        super().save(filepath)

    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return create_default_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy="oversample",
            classifier=XGBClassifier,
            classifier_params=fixed_params
        )
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_.best_estimator_
        



class MyESXGBClassifier(AbstractEstimator):
    '''
    Class that wraps the XGBClassifier used with early stopping
    and without HPs tuning. We use the default xgboost parameters
    expect for the trees number.

    Attributes
    ------------------    
    estimator_ (XGBClassifier): 
        Fitted XGBClassifier.
    preprocessing_pipeline_ (Pipeline): 
        Fitted Pipeline that applies the preprocessing.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions = None,
        fixed_params: dict = ES_XGBCLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyESXGBClassifier":
        X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            train_size=0.75, 
            random_state=self.seed, 
            stratify=y
        )
        self.preprocessing_pipeline_ = self._create_preprocessing_pipeline()
        X_train_trans = self.preprocessing_pipeline_.fit_transform(X_train)
        X_val_trans = self.preprocessing_pipeline_.transform(X_val)

        fixed_params = _adjust_xgb_objective(self.fixed_params, y)
        fixed_params = _adjust_logloss_es_metric(fixed_params, y)
        estimator = XGBClassifier(**fixed_params)
        
        self.estimator_ = estimator.fit(
            X_train_trans, 
            y_train, 
            eval_set=[(X_val_trans, y_val)], 
            verbose=False
        ) 
        return self
    
    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()
    
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocessing_pipeline_.transform(X)
        return super()._classic_predict_proba(X)
    
    def save(self, filepath: str | Path) -> None:
        super().save(filepath)
    
    def _create_preprocessing_pipeline(self) -> Pipeline:
        return create_default_pipeline(self.preprocessing, "oversample")
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.preprocessing_pipeline_




class MyXGBClassifier(AbstractEstimator):
    '''
    Class that wraps the XGBClassifier used without HPs tuning.
    We use the dafault xgboost parameters expect for the trees number.

    Attributes
    ------------------    
    estimator_ (Pipeline): Fitted Pipeline with XGBClassifier as last step. 
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions = None,
        fixed_params: dict = XGBCLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, params_distributions, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyXGBClassifier":
        fixed_params = _adjust_xgb_objective(self.fixed_params, y)
        estimator = self._create_estimator(fixed_params)
        self.estimator_ = estimator.fit(X, y)
        return self
    
    def collect_fit_preprocessing_info(self) -> dict:
        return super().collect_fit_preprocessing_info()

    def get_feature_names_in_(self) -> np.ndarray:
        return super().get_feature_names_in_()

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return super()._classic_predict_proba(X)
    
    def save(self, filepath: str | Path) -> None:
        super().save(filepath)

    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return create_default_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy="oversample",
            classifier=XGBClassifier,
            classifier_params=fixed_params
        )
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_




def _adjust_xgb_objective(fixed_params: dict, y: pd.Series) -> dict:
    '''
    Add the objective parameter since it is fit/data specific.
    Returns a new dict.
    '''
    copy_fixed_params = deepcopy(fixed_params)
    n_classes = y.unique().size
    
    if n_classes == 2:
        # here we must NOT specify num_class otherwise strange behaviours
        copy_fixed_params["objective"] = "binary:logistic"
    else:
        copy_fixed_params["objective"] = "multi:softprob"
        copy_fixed_params["num_class"] = n_classes
    
    return copy_fixed_params 



def _adjust_logloss_es_metric(fixed_params: dict, y: pd.Series) -> dict:
    '''
    The XGBboost package differentiate between binary "logloss"
    and multiclassification "mlogloss" for the early stopping metric.
    Returns a new dict of fixed params with the correct logloss if used.
    '''
    fixed_params = deepcopy(fixed_params)
    is_log_loss = True if fixed_params["eval_metric"] in ["logloss", "mlogloss"] else False

    if not is_log_loss:
        return fixed_params
    
    n_classes = y.unique().size
    
    if n_classes == 2:
        fixed_params["eval_metric"] = "logloss"
    else:
        fixed_params["eval_metric"] = "mlogloss"
    
    return fixed_params