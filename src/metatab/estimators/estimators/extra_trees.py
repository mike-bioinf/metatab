from sklearn.ensemble import ExtraTreesClassifier
from metatab.estimators.params import TuningParams, DefaultParams
from metatab.metatab_utils.types import XType, YType

from metatab.estimators.core import (
    AbstractBaseEstimator, 
    DefaultEstimatorMixin,
    TunedEstimatorMixin,
    EnsembleEstimatorMixin,
    MetaTuneBaseEstimator,
    MetaEnsembleBaseEstimator
)



class MyExtraTreesClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library ExtraTreesClassifier.

    Attributes:
        estimator_ (ExtraTreesClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params = DefaultParams.EXTRA_TREES_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyExtraTreesClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=ExtraTreesClassifier,
            type_estimator="extra_trees"
        )
        return self
    


class MyTunedExtraTreesClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned ExtraTreesClassifier.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.    
    '''
    fixed_params = TuningParams.EXTRA_TREES_FIXED_PARAMS 
    
    def fit(self, X: XType, y: YType) -> "MyTunedExtraTreesClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=ExtraTreesClassifier,
            type_estimator="extra_trees",
            is_tuned=True
        )
        return self
    


class MyEnsembledExtraTreesClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of ensemble ExtraTreesClassifier
    
    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params = TuningParams.EXTRA_TREES_FIXED_PARAMS
    
    def fit(self, X: XType, y: YType) -> "MyEnsembledExtraTreesClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=ExtraTreesClassifier,
            type_estimator="extra_trees",
            is_ensembled=True
        )
        return self
    


class MetaTuneExtraTreesClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneExtraTreesClassifier":
        super().fit(X, y, "base", MyTunedExtraTreesClassifier, TuningParams.EXTRA_TREES_C0, None)
        return self



class MetaEnsembleExtraTreesClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaEnsembleExtraTreesClassifier":
        super().fit(X, y, "base", MyEnsembledExtraTreesClassifier, TuningParams.EXTRA_TREES_C0, None)
        return self