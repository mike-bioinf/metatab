import warnings
from functools import partial
from lightgbm import LGBMClassifier
from estimators.params import TuningParams, DefaultParams
from estimators.core.configurations import EarlyStopConfiguration
from metatab_utils.types import XType, YType

from estimators.core import (
    AbstractBaseEstimator, 
    DefaultEstimatorMixin,
    TunedEstimatorMixin, 
    EnsembleEstimatorMixin,
    MetaTuneBaseEstimator,
    MetaEnsembleBaseEstimator
)

from estimators.utils.gbdt import ( 
    adjust_objective_logloss_and_num_classes,
    adjust_es_logloss_metric
)



## We have to use this decorator on predict methods of all LGBM classes and 
## on the fit of the tuned classes since prediction is performed in cross validation
def ignore_lgbm_feature_name_warning(method):
    '''
    Method decorator to filter the warning "X does not have valid feature names"
    raising from a bug in lgbm that checks at predict level the learned 
    artifical column names that it gives to numpy arrays at fit level.
    github issue: "https://github.com/microsoft/LightGBM/issues/6798".
    '''
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "X does not have valid feature names.*")
            return method(*args, **kwargs)
    return wrapper



class MyLGBMClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of default library LGBMClassifier without tuning and early stop.

    Attributes:
        estimator_ (LGBMClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params=DefaultParams.LGBM_DEFAULT_PARAMS

    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="lgbm",
            is_tuned=False,
            is_early_stopped=False,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm")
            ]
        )
        return self

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)



class MyESLGBMClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library LGBMClassifier with early stop and without tuning. 
    
    Attributes:
        estimator_ (LGBMClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params=DefaultParams.ES_LGBM_DEFAULT_PARAMS

    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyESLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="es_lgbm",
            is_tuned=False,
            is_early_stopped=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm"),
                partial(adjust_es_logloss_metric, framework="lightgbm")
            ]
        )
        return self

    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)



class MyTunedLGBMClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned LGBMClassifier without early stop.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.    
    '''
    fixed_params=TuningParams.LGBM_FIXED_PARAMS
        
    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyTunedLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="lgbm",
            is_tuned=True,
            is_early_stopped=False,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm")
            ]
        )
        return self
    
    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)



class MyTunedESLGBMClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned LGBMClassifier with early stop.
    
    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params = TuningParams.ES_LGBM_FIXED_PARAMS

    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyTunedESLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="es_lgbm",
            is_tuned=True,
            is_early_stopped=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm"),
                partial(adjust_es_logloss_metric, framework="lightgbm") 
            ]
        )
        return self
    
    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)
 


class MyEnsembledLGBMClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the ensmebled LGBMClassifier without early stop.

    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.    
    '''
    fixed_params=TuningParams.LGBM_FIXED_PARAMS
        
    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyEnsembledLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="lgbm",
            is_ensembled=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm")
            ]
        )
        return self
    
    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)   



class MyEnsembledESLGBMClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the ensmebled LGBMClassifier with early stop.
    
    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params = TuningParams.ES_LGBM_FIXED_PARAMS

    @ignore_lgbm_feature_name_warning
    def fit(self, X: XType, y: YType) -> "MyEnsembledESLGBMClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=LGBMClassifier,
            type_estimator="es_lgbm",
            is_ensembled=True,
            is_early_stopped=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="lightgbm"),
                partial(adjust_es_logloss_metric, framework="lightgbm") 
            ]
        )
        return self
    
    @ignore_lgbm_feature_name_warning
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X)



class MetaTuneLGBMClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneLGBMClassifier":
        super().fit(X, y, "base", MyTunedLGBMClassifier, TuningParams.LGMB_C0, None)
        return self



class MetaTuneEsLGBMClassifier(MetaTuneBaseEstimator):
    def fit(
        self,
        X: XType, 
        y: YType, 
        early_stop_rounds: int = 100, 
        validation_set_size: float = 0.3
    ) -> "MetaTuneEsLGBMClassifier":
        early_stop_conf = EarlyStopConfiguration(early_stop_rounds, validation_set_size)
        super().fit(X, y, "base", MyTunedESLGBMClassifier, TuningParams.LGMB_C0, early_stop_conf)
        return self
    


class MetaEnsembleLGBMClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaEnsembleLGBMClassifier":
        super().fit(X, y, "base", MyEnsembledLGBMClassifier, TuningParams.LGMB_C0, None)
        return self
    


class MetaEnsembleEsLGBMClassifier(MetaEnsembleBaseEstimator):
    def fit(
        self,
        X: XType, 
        y: YType, 
        early_stop_rounds: int = 100, 
        validation_set_size: float = 0.3
    ) -> "MetaEnsembleEsLGBMClassifier":
        early_stop_conf = EarlyStopConfiguration(early_stop_rounds, validation_set_size)
        super().fit(X, y, "base", MyEnsembledESLGBMClassifier, TuningParams.LGMB_C0, early_stop_conf)
        return self