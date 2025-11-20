from functools import partial
from xgboost import XGBClassifier
from estimators.params import TuningParams, DefaultParams
from estimators.core.configurations import EarlyStopConfiguration
from metatab_utils.types import XType, YType

from estimators.core import (
    AbstractBaseEstimator, 
    TunedEstimatorMixin, 
    DefaultEstimatorMixin,
    BaseMetaEstimator
)

from estimators.utils.gbdt import ( 
    adjust_objective_logloss_and_num_classes,
    adjust_es_logloss_metric
)



class MyXGBClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default XGBClassifier without tuning and early stop.

    Attributes:
        estimator_ (XGBClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params = DefaultParams.XGB_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyXGBClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=XGBClassifier,
            type_estimator="xgb",
            is_early_stopped=False,
            is_tuned=False,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="xgboost")
            ]
        )
        return self



class MyESXGBClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library XGBClassifier with early stop and without tuning.

    Attributes:
        estimator_ (XGBClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params=DefaultParams.ES_XGB_DEFAULT_PARAMS
    
    def fit(self, X: XType, y: YType) -> "MyESXGBClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=XGBClassifier,
            type_estimator="es_xgb",
            is_tuned=False,
            is_early_stopped=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="xgboost"),
                partial(adjust_es_logloss_metric, framework="xgboost")
            ],
            fit_classifier_kwargs={"verbose": False} # to be effective must be passed to fit
        )
        return self



class MyTunedXGBClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned XGBClassifier without early stop.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params=TuningParams.XGB_FIXED_PARAMS

    def fit(self, X: XType, y: YType) -> "MyTunedXGBClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=XGBClassifier,
            type_estimator="xgb",
            is_early_stopped=False,
            is_tuned=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="xgboost")
            ]
        )
        return self



class MyTunedESXGBClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned XGBClassifier with early stop.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params = TuningParams.ES_XGB_FIXED_PARAMS

    def fit(self, X: XType, y: YType) -> "MyTunedESXGBClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=XGBClassifier,
            type_estimator="es_xgb",
            is_early_stopped=True,
            is_tuned=True,
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="xgboost"),
                partial(adjust_es_logloss_metric, framework="xgboost")
            ],
            fit_classifier_kwargs={"verbose": False}  # to be effective must be passed to fit
        )
        return self
    


class MetaTuneXGBClassifier(BaseMetaEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneXGBClassifier":
        super().fit(X, y, "base", MyTunedXGBClassifier, TuningParams.XGB_C0, None)
        return self



class MetaTuneEsXGBClassifier(BaseMetaEstimator):
    def fit(
        self,
        X: XType, 
        y: YType,
        early_stop_rounds: int = 100,
        validation_set_size: float = 0.3
    ) -> "MetaTuneEsXGBClassifier":
        early_stop_conf = EarlyStopConfiguration(
            early_stop_rounds=early_stop_rounds, 
            validation_set_size=validation_set_size
        )
        super().fit(X, y, "base", MyTunedESXGBClassifier, TuningParams.XGB_C0, early_stop_conf)
        return self