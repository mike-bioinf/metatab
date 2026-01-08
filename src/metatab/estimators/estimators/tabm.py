from __future__ import annotations

import warnings
import pandas as pd
from typing import TYPE_CHECKING, Literal
from sklearn.utils.validation import check_is_fitted
from pytabkit import TabM_D_Classifier
from metatab.estimators.params import DefaultParams, TuningParams
from metatab.estimators.core.configurations import EarlyStopConfiguration

from metatab.estimators.core import (
    AbstractBaseEstimator,
    DefaultEstimatorMixin,
    EnsembleEstimatorMixin,
    TunedEstimatorMixin,
    MetaTuneBaseEstimator,
    MetaEnsembleBaseEstimator
)

if TYPE_CHECKING:
    import numpy as np
    from metatab.metatab_utils.types import XType, YType



def suppress_tabm_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
                category=FutureWarning
            )
            warnings.filterwarnings(
                action="ignore",
                message="The.*feature has just two bin edges, which means only one bin.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper



# README: 
# The interface does not follow the sklearn design using **params in init.
# For this reason we miss some sklearn features and compabilities.
# These should not be used/useful to us (the useful ones are explicitely implemented). 
class TabMClassifier:
    '''
    Wrapper of pytabkit "TabM_D_Classifier" that allows for:
    1. "auto" batch size option.
    2. Suppression of undesired warnings.
    3. fit method with eval_set-like interface.
    4. learns "feature_names_in_" attribute.

    Parameters:
        Accept all TabM_D_Classifier parameters.
    '''
    def __init__(self, batch_size: int | Literal["auto"] = 256, **params):
        self.batch_size=batch_size
        self.params=params


    @suppress_tabm_warnings
    def fit(self, X: XType, y: YType, eval_set: list[tuple[XType, YType]], **kwargs) -> "TabMClassifier":
        '''
        Fit interface accepting the X and y validation sets via "eval_set-like" parameter.
        Accepts the additional tabm args via kwargs.

        Parameters:
            X (XType): data to fit.
            y (YType): data labels to fit.
            eval_set (list[tuple[XType, YType]]):
                List of a SINGLE binary tuple with the Xy validation sets.
                Here we use this signature to be consistent with the GBDTs
                interfaces which accept multiple (potentially) validation sets.
                Cannot be None since we do NOT want to rely on the internal automatic
                system to infer the validation sets. Infact they are always used by TabM_D_Classifier.
            Kwargs:
                Additional keyword parameters accepted by the fit method of TabM_D_Classifier.
                The arguments that pass/specify the validation sets must not be used in favor of
                the `eval_set` parameter.
        '''
        from metatab.estimators.utils.general import collect_sklearn_classification_fit_info

        self.batch_size_ = self.batch_size \
            if isinstance(self.batch_size, int) \
            else self._infer_batch_size_from_data(X)
        
        X_val = eval_set[0][0]
        y_val = eval_set[0][1]
        self.classifier = TabM_D_Classifier(batch_size=self.batch_size_, **self.params)
        self.classifier.fit(X, y, X_val=X_val, y_val=y_val, **kwargs)
        
        for k, v in collect_sklearn_classification_fit_info(self.classifier).items():
            setattr(self, k, v)

        # learn feature_names_in_ which is not learned by pytabkit classifiers
        if isinstance(X, pd.DataFrame) and all([isinstance(col, str) for col in X.columns]):
            self.feature_names_in_ = X.columns
        
        return self
    

    @suppress_tabm_warnings
    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "classifier")
        return self.classifier.predict(X)
    

    @suppress_tabm_warnings
    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "classifier")
        return self.classifier.predict_proba(X)
    

    def set_params(self, **params) -> "TabMClassifier":
        if params.get("batch_size", None) is not None:
            self.batch_size = params.pop("batch_size")
        # here we must update to not lose all previous info
        self.params.update(params)
        return self
    

    def get_params(self, *args, **kwargs) -> dict:
        return {"batch_size": self.batch_size, **self.params}


    @staticmethod
    def _infer_batch_size_from_data(X: XType) -> None:
        n_samples = X.shape[0]
        if n_samples <= 256*3:
            return int(n_samples / 3)
        else:
            # the default
            return 256



class MyTabMClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the library (pytabkit) default TabMClassifier.

    Attributes:
        estimator_ (Pipeline): Fitted pipeline object.
    '''
    fixed_params = DefaultParams.TABM_DEFAULT_PARAMS
 
    def fit(self, X: XType, y: YType) -> "MyTabMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabMClassifier,
            type_estimator="tabm",
            is_early_stopped=True,
            early_stop_rounds_parameter=None, # we use fixed patience
            n_threads_parameter="n_threads",
            device_parameter="device",
            density_feature_selector_strategy="undersample" # to speed up
        )
        return self
    


class MyTunedTabMClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned TabMClassifier.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params = TuningParams.TABM_C0

    def fit(self, X: XType, y: YType) -> "MyTunedTabMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabMClassifier,
            type_estimator="tabm",
            is_tuned=True,
            is_early_stopped=True,
            early_stop_rounds_parameter=None, # we use fixed patience
            n_threads_parameter="n_threads",
            device_parameter="device",
            density_feature_selector_strategy="undersample" # to speed up.
        )
        return self
    


class MyEnsembledTabMClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the ensembled TabMClassifier.

    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params = TuningParams.TABM_C0

    def fit(self, X: XType, y: YType) -> "MyEnsembledTabMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabMClassifier,
            type_estimator="tabm",
            is_ensembled=True,
            is_early_stopped=True,
            early_stop_rounds_parameter=None, # we use fixed patience
            n_threads_parameter="n_threads",
            device_parameter="device",
            density_feature_selector_strategy="undersample" # to speed up.
        )
        return self
    


class MetaTuneTabMClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType, validation_set_size: float = 0.3) -> "MetaTuneTabMClassifier":
        esc = EarlyStopConfiguration(validation_set_size=validation_set_size)
        super().fit(X, y, "base", MyTunedTabMClassifier, TuningParams.TABM_C0, esc)
        return self



class MetaEnsembleTabMClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y:YType, validation_set_size: float = 0.3) -> "MetaEnsembleTabMClassifier":
        esc = EarlyStopConfiguration(validation_set_size=validation_set_size)
        super().fit(X, y, "base", MyEnsembledTabMClassifier, TuningParams.TABM_C0, esc)
        return self