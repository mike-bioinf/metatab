from sklearn.ensemble import RandomForestClassifier
from metatab.estimators.params import TuningParams, DefaultParams
from metatab.metatab_utils.types import XType, YType

from metatab.estimators.core import (
    AbstractBaseEstimator, 
    DefaultEstimatorMixin,
    TunedEstimatorMixin,
    EnsembleEstimatorMixin,
    MetaTuneBaseEstimator
)

from metatab.estimators.core.meta_ens_base_estimator import (
    MetaEnsembleInitializer, 
    RandomEnsembleInitializer,
    BaseEnsembleEstimator
)



class MyRandomForestClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library RandomForestClassifier.

    Attributes:
        estimator_ (Pipeline): Fitted pipeline object.
    '''
    fixed_params = DefaultParams.RANDOM_FOREST_DEFAULT_PARAMS

    def fit(self, X: XType, y: YType) -> "MyRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest"
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










class MetaEnsembleRandomForestClassifier(MetaEnsembleInitializer, BaseEnsembleEstimator):     
    def fit(self, X: XType, y: YType) -> "MetaEnsembleRandomForestClassifier":        
        super().fit(
            X=X, 
            y=y, 
            classifier_cls=RandomForestClassifier,
            classifier_random_state_parameter="random_state",
            classifier_nthreads_paramater="n_jobs",
            classifier_device_parameter=None,
            fixed_params=TuningParams.RANDOM_FOREST_FIXED_PARAMS, 
            tuning_params=TuningParams.RF_C0, 
            type_estimator="random_forest",
        )
        return self