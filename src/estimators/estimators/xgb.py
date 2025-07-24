import numpy as np
import pandas as pd
from typing import Literal, override
from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import ( 
    RandomizedSearchCV, 
    train_test_split
)

from sklearn.utils.validation import check_is_fitted
from estimators.estimators.random_search import MyRandomSearchCV
from estimators.estimators.abstract_estimator import AbstractBaseEstimator

from estimators.estimators.utils import (
    add_string_to_params, 
    create_default_pipeline,
    remove_string_from_params
)

from estimators.estimators.params import ( 
    ES_XGBCLASSIFIER_FIXED_PARAMS,
    XGBCLASSIFIER_FIXED_PARAMS,
    SKLEARN_RANDOM_SEARCH_FIXED_PARAMS
)



class MyRandomizedESXGBClassifier(AbstractBaseEstimator):
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
        n_cores: int, 
        tune_configuration: dict,
        fixed_params: dict = ES_XGBCLASSIFIER_FIXED_PARAMS
    ):
        super().__init__(preprocessing, seed, n_cores, tune_configuration, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomizedESXGBClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_cores=True, copy=True)
        fixed_params = _adjust_xgb_objective(fixed_params, y)
        fixed_params = _adjust_logloss_es_metric(fixed_params, y)
       
        estimator = MyRandomSearchCV(
            classifier=XGBClassifier,
            fixed_params_classifier=fixed_params,
            param_distributions=self.tune_configuration["params_distributions"],
            preprocessing_pipeline=self._create_preprocessing_pipeline(),
            splitter=super().build_tune_splitter(),
            scorer="logloss",
            n_iter=self.tune_configuration["n_iter"],
            refit=True,
            seed=self.seed
        )

        self.estimator_ = estimator.fit(X, y)
        return self     

    def _create_preprocessing_pipeline(self) -> Pipeline:
        return create_default_pipeline(self.preprocessing, "oversample")

    def _get_fitted_preprocessing_pipeline_or_estimator(self):
        return self.estimator_.preprocessing_pipeline_
    
    @override
    def get_best_hps(self) -> dict:
        check_is_fitted(self, "estimator_")
        return self.estimator_.best_params_




class MyRandomizedXGBClassifier(AbstractBaseEstimator):
    '''
    Class that implements the xgboost classifier in random search
    to tune the HPs without early stopping.
    
    Attributes
    ------------------    
    estimator_ (RandomizedSearchCV): 
        Fitted RandomizedSearchCV on a Pipeline with XGBClassifier as last step.
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_cores: int,
        tune_configuration: dict,
        fixed_params: dict = XGBCLASSIFIER_FIXED_PARAMS  
    ):
        super().__init__(preprocessing, seed, n_cores, tune_configuration, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyRandomizedXGBClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_cores=True, copy=True)
        fixed_params = _adjust_xgb_objective(fixed_params, y)
        
        params_distributions = add_string_to_params(
            self.tune_configuration["params_distributions"], 
            "xgbclassifier__"
        )

        estimator = RandomizedSearchCV(
            estimator=self._create_estimator(fixed_params),
            param_distributions=params_distributions,
            n_iter=self.tune_configuration["n_iter"],
            cv=super().build_tune_splitter(),
            random_state=self.seed,
            **SKLEARN_RANDOM_SEARCH_FIXED_PARAMS,
        )

        self.estimator_ = estimator.fit(X, y)
        return self

    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return create_default_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy="oversample",
            classifier=XGBClassifier,
            classifier_params=fixed_params
        )
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_.best_estimator_
    
    @override
    def get_best_hps(self) -> dict:
        check_is_fitted(self, "estimator_")
        return remove_string_from_params(self.estimator_.best_params_, "xgbclassifier__")




class MyESXGBClassifier(AbstractBaseEstimator):
    '''
    Class that wraps the XGBClassifier used with early stopping
    and without HPs tuning. We use the default xgboost parameters
    exept for the trees number.

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
        n_cores: int,
        tune_configuration = None,
        fixed_params: dict = ES_XGBCLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, n_cores, tune_configuration, fixed_params)

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

        fixed_params = super().update_fixed_params(up_seed=True, up_n_cores=True, copy=True)
        fixed_params = _adjust_xgb_objective(fixed_params, y)
        fixed_params = _adjust_logloss_es_metric(fixed_params, y)
        estimator = XGBClassifier(**fixed_params)
        
        self.estimator_ = estimator.fit(
            X_train_trans, 
            y_train, 
            eval_set=[(X_val_trans, y_val)], 
            verbose=False
        )

        return self
    
    @override
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        X = self.preprocessing_pipeline_.transform(X)
        return self.estimator_.predict_proba(X)
    
    def _create_preprocessing_pipeline(self) -> Pipeline:
        return create_default_pipeline(self.preprocessing, "oversample")
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.preprocessing_pipeline_




class MyXGBClassifier(AbstractBaseEstimator):
    '''
    Class that wraps the XGBClassifier used without HPs tuning.
    We use the dafault xgboost parameters exept for the trees number.

    Attributes
    ------------------    
    estimator_ (Pipeline): Fitted Pipeline with XGBClassifier as last step. 
    '''
    def __init__(
        self,
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        n_cores: int,
        tune_configuration = None,
        fixed_params: dict = XGBCLASSIFIER_FIXED_PARAMS   
    ):
        super().__init__(preprocessing, seed, n_cores, tune_configuration, fixed_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MyXGBClassifier":
        fixed_params = super().update_fixed_params(up_seed=True, up_n_cores=True, copy=True)
        fixed_params = _adjust_xgb_objective(fixed_params, y)
        estimator = self._create_estimator(fixed_params)
        self.estimator_ = estimator.fit(X, y)
        return self

    def _create_estimator(self, fixed_params: dict) -> Pipeline:
        return create_default_pipeline(
            preprocessing=self.preprocessing,
            density_feature_selector_strategy="oversample",
            classifier=XGBClassifier,
            classifier_params=fixed_params
        )
    
    def _get_fitted_preprocessing_pipeline_or_estimator(self) -> Pipeline:
        return self.estimator_




## TODO: merge the 2 adapt funcs into one
def _adjust_xgb_objective(fixed_params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    Add the objective parameter since it is fit/data specific.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    fixed_params = deepcopy(fixed_params) if copy else fixed_params
    n_classes = y.unique().size
    
    if n_classes == 2:
        # here we must NOT specify num_class otherwise strange behaviours
        fixed_params["objective"] = "binary:logistic"
    else:
        fixed_params["objective"] = "multi:softprob"
        fixed_params["num_class"] = n_classes
    
    return fixed_params 



def _adjust_logloss_es_metric(fixed_params: dict, y: pd.Series, copy: bool = False) -> dict:
    '''
    The XGBboost package differentiate between binary "logloss"
    and multiclassification "mlogloss" for the early stopping metric.
    The function adjust the logloss according to the classification scenario,
    if it used for early stop.
    Returns a new dict or the old updated one depending on copy parameter.
    '''
    fixed_params = deepcopy(fixed_params) if copy else fixed_params
    is_log_loss = True if fixed_params["eval_metric"] in ["logloss", "mlogloss"] else False

    if not is_log_loss:
        return fixed_params
    
    n_classes = y.unique().size
    
    if n_classes == 2:
        fixed_params["eval_metric"] = "logloss"
    else:
        fixed_params["eval_metric"] = "mlogloss"
    
    return fixed_params