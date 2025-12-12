from sklearn.ensemble import RandomForestClassifier
from estimators.params import TuningParams, DefaultParams
from metatab_utils.types import XType, YType

from estimators.core import (
    AbstractBaseEstimator, 
    DefaultEstimatorMixin,
    TunedEstimatorMixin,
    EnsembleEstimatorMixin,
    MetaTuneBaseEstimator,
    MetaEnsembleBaseEstimator
)



class MyRandomForestClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library RandomForestClassifier.

    Attributes:
        estimator_ (RandomForestClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params = DefaultParams.RANDOM_FOREST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest",
            is_tuned=False
        )
        return self
       


class MyTunedRandomForestClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned RandomForestClassifier.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.    
    '''
    fixed_params = TuningParams.RANDOM_FOREST_FIXED_PARAMS 
    
    def fit(self, X: XType, y: YType) -> "MyTunedRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest",
            is_tuned=True
        )
        return self



class MyEnsembledRandomForestClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of ensemble RandomForestClassifier
    
    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params = TuningParams.RANDOM_FOREST_FIXED_PARAMS
    
    def fit(self, X: XType, y: YType) -> "MyEnsembledRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest",
            is_ensembled=True
        )
        return self



class MetaTuneRandomForestClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneRandomForestClassifier":
        super().fit(X, y, "base", MyTunedRandomForestClassifier, TuningParams.RF_C0, None)
        return self



class MetaEnsembleRandomForestClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaEnsembleRandomForestClassifier":
        super().fit(X, y, "base", MyEnsembledRandomForestClassifier, TuningParams.RF_C0, None)
        return self