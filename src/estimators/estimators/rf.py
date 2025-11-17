import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from estimators.params import TuningParams, DefaultParams

from estimators.core import (
    AbstractBaseEstimator, 
    DefaultEstimatorMixin,
    TunedEstimatorMixin
)



class MyRandomForestClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the default library RandomForestClassifier.

    Attributes:
        estimator_ (RandomForestClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params = DefaultParams.RANDOM_FOREST_DEFAULT_PARAMS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest",
            is_tuned=False,
            is_early_stopped=False
        )
        return self
       


class MyTunedRandomForestClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned RandomForestClassifier.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.    
    '''
    fixed_params = TuningParams.RANDOM_FOREST_FIXED_PARAMS 
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTunedRandomForestClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=RandomForestClassifier,
            type_estimator="random_forest",
            is_tuned=True,
            is_early_stopped=False
        )
        return self