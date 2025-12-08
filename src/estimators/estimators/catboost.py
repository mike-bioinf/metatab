from functools import partial
from catboost import CatBoostClassifier
from estimators.params import TuningParams, DefaultParams
from metatab_utils.types import XType, YType

from estimators.core import (
    DefaultEstimatorMixin,
    TunedEstimatorMixin,
    EnsembleEstimatorMixin,
    AbstractBaseEstimator
)

from estimators.utils.gbdt import (
    adjust_es_logloss_metric,
    adjust_objective_logloss_and_num_classes
)



class MyCatBoostClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of library default CatBoostClassifier without early stop.

    Attributes:
        estimator_ (CatBoostClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params=DefaultParams.CATBOOST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="catboost",
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost")
            ]
        )
        return self



class MyESCatBoostClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the library default CatBoostClassifier with early stop.

    Attributes:
        estimator_ (CatBoostClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params=DefaultParams.ES_CATBOOST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyESCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="es_catboost",
            is_early_stopped=True,
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost"),
                partial(adjust_es_logloss_metric, framework="catboost")
            ]
        )
        return self



class MyTunedCatBoostClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of tuned CatBoostClassifier without early stop.
    
    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params=TuningParams.CATBOOST_FIXED_PARAMS

    def fit(self, X: XType, y: YType) -> "MyTunedCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="catboost",
            is_tuned=True,
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost")
            ]
        )
        return self



class MyTunedESCatBoostClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of tuned CatBoostClassifier with early stop.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params=TuningParams.ES_CATBOOST_FIXED_PARAMS

    def fit(self, X: XType, y: YType) -> "MyTunedESCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="es_catboost",
            is_tuned=True,
            is_early_stopped=True,
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost"),
                partial(adjust_es_logloss_metric, framework="catboost")
            ]
        )
        return self
    


class MyEnsembledCatBoostClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of ensembled CatBoostClassifier.
    
    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params=TuningParams.CATBOOST_FIXED_PARAMS
    
    def fit(self, X: XType, y: YType) -> "MyEnsembledCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="catboost",
            is_ensembled=True,
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost")
            ]
        )
        return self



class MyEnsembledESCatBoostClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of ensembled CatBoostClassifier with early stop.
    
    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params=TuningParams.ES_CATBOOST_FIXED_PARAMS

    def fit(self, X: XType, y: YType) -> "MyEnsembledESCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifier,
            type_estimator="es_catboost",
            is_ensembled=True,
            is_early_stopped=True,
            n_threads_parameter="thread_count",
            callbacks_on_fixed_params=[
                partial(adjust_objective_logloss_and_num_classes, framework="catboost"),
                partial(adjust_es_logloss_metric, framework="catboost")
            ]
        )
        return self