import pandas as pd
from functools import partial
from catboost import CatBoostClassifier
from metatab.estimators.params import TuningParams, DefaultParams
from metatab.metatab_utils.types import XType, YType
from metatab.estimators.core.configurations import EarlyStopConfiguration

from metatab.estimators.core import (
    DefaultEstimatorMixin,
    TunedEstimatorMixin,
    EnsembleEstimatorMixin,
    MetaEnsembleBaseEstimator,
    MetaTuneBaseEstimator,
    AbstractBaseEstimator
)

from metatab.estimators.utils.gbdt import (
    adjust_es_logloss_metric,
    adjust_objective_logloss_and_num_classes
)



class CatBoostClassifierInterface(CatBoostClassifier):
    '''
    Interface of "CatboostClassifier" that allows to learn the `feature_names_in_` attribute. 
    The original classifier infact uses `feature_names_`, which is learned even when we fit
    the classifier on numpy arrays.
    '''
    def fit(self, X: XType, y: YType, **kwargs) -> "CatBoostClassifierInterface":
        super().fit(X, y, **kwargs)
        if isinstance(X, pd.DataFrame) and all([isinstance(col, str) for col in X.columns]):
            self.feature_names_in_ = self.feature_names_
        return self



class MyCatBoostClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of library default CatBoostClassifier without early stop.

    Attributes:
        estimator_ (Pipeline): Fitted pipeline object.
    '''
    fixed_params=DefaultParams.CATBOOST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifierInterface,
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
        estimator_ (Pipeline): Fitted pipeline object.
    '''
    fixed_params=DefaultParams.ES_CATBOOST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyESCatBoostClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=CatBoostClassifierInterface,
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
            classifier_cls=CatBoostClassifierInterface,
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
            classifier_cls=CatBoostClassifierInterface,
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
            classifier_cls=CatBoostClassifierInterface,
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
            classifier_cls=CatBoostClassifierInterface,
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
    


class MetaTuneCatBoostClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneCatBoostClassifier":
        super().fit(X, y, "base", MyTunedCatBoostClassifier, TuningParams.CATBOOST_C0, None)
        return self



class MetaTuneEsCatBoostClassifier(MetaTuneBaseEstimator):
    def fit(
        self,
        X: XType, 
        y: YType,
        early_stop_rounds: int = 100,
        validation_set_size: float = 0.3
    ) -> "MetaTuneEsCatBoostClassifier":
        early_stop_conf = EarlyStopConfiguration(early_stop_rounds, validation_set_size)
        super().fit(X, y, "base", MyTunedESCatBoostClassifier, TuningParams.CATBOOST_C0, early_stop_conf)
        return self



class MetaEnsembleCatboostClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaEnsembleCatboostClassifier":
        super().fit(X, y, "base", MyEnsembledCatBoostClassifier, TuningParams.CATBOOST_C0, None)
        return self
    


class MetaEnsembleEsCatboostClassifier(MetaEnsembleBaseEstimator):
    def fit(
        self, 
        X: XType, 
        y: YType, 
        early_stop_rounds: int = 100, 
        validation_set_size: float = 0.3
    ) -> "MetaEnsembleEsCatboostClassifier":
        early_stop_conf = EarlyStopConfiguration(early_stop_rounds, validation_set_size)
        super().fit(X, y, "base", MyEnsembledESCatBoostClassifier, TuningParams.CATBOOST_C0, early_stop_conf)
        return self