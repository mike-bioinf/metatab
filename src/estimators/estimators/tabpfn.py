import warnings
from tabpfn import TabPFNClassifier
from estimators.params import DefaultParams, TuningParams
from hp_search.tabpfn_search_space import download_and_return_tabpfn_checkpoints, TABPFN_CHECKPOINTS
from metatab_utils.types import XType, YType

from estimators.core import (
    AbstractBaseEstimator,
    DefaultEstimatorMixin,
    EnsembleEstimatorMixin,
    TunedEstimatorMixin,
    MetaTuneBaseEstimator,
    MetaEnsembleBaseEstimator
)



## TODO: to update when updating tabpfn version:
# you can probably remove the sklearn warnings, and update the number of features in the message.
def suppress_sklearn_and_tabpfn_warnings(func):
    '''
    Decorator to filter sklearn future deprecation warnings,
    and tabpfn loading and ignore training limits warning.
    '''
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*", module=".*tabpfn.*loading")
            warnings.filterwarnings(
                action="ignore", 
                message=".*is greater than the maximum Number of features 500 supported by the model.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper



class MyTabPFNClassifier(DefaultEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the library default TabPFNClassifier.

    Attributes:
        estimator_ (TabPFNClassifier|Pipeline): Fitted classifier or pipeline object.
    '''
    fixed_params = DefaultParams.TABPFN_DEFAULT_PARAMS
 
    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: XType, y: YType) -> "MyTabPFNClassifier":
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabPFNClassifier,
            type_estimator="tabpfn",
            is_tuned=False,
            is_early_stopped=False,
            density_feature_selector_strategy="undersample" # to speed up and be consistent with the tuned estimator
        )
        return self
    


class MyTunedTabPFNClassifier(TunedEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the tuned TabPFNClassifier.

    Attributes:
        estimator_ (SearchCV): Fitted SearchCV object.
    '''
    fixed_params = TuningParams.TABPFN_FIXED_PARAMS

    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: XType, y: YType) -> "MyTunedTabPFNClassifier":
        # we download the different tabpfn checkpoint in the user cache dir
        _ = download_and_return_tabpfn_checkpoints(TABPFN_CHECKPOINTS)
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabPFNClassifier,
            type_estimator="tabpfn",
            is_tuned=True,
            is_early_stopped=False,
            density_feature_selector_strategy="undersample" # to speed up.
        )
        return self



class MyEnsembledTabPFNClassifier(EnsembleEstimatorMixin, AbstractBaseEstimator):
    '''
    Implementation of the ensembled TabPFNClassifier.

    Attributes:
        estimator_ (EnsembleEstimator): Fitted EnsembleEstimator object.
    '''
    fixed_params = TuningParams.TABPFN_FIXED_PARAMS

    @suppress_sklearn_and_tabpfn_warnings
    def fit(self, X: XType, y: YType) -> "MyTunedTabPFNClassifier":
        # we download the different tabpfn checkpoint in the user cache dir
        _ = download_and_return_tabpfn_checkpoints(TABPFN_CHECKPOINTS)
        self.estimator_ = super().fit_estimator(
            X=X,
            y=y,
            classifier_cls=TabPFNClassifier,
            type_estimator="tabpfn",
            is_ensembled=True,
            is_early_stopped=False,
            density_feature_selector_strategy="undersample" # to speed up.
        )
        return self



class MetaTuneTabPFNClassifier(MetaTuneBaseEstimator):
    def fit(self, X: XType, y: YType) -> "MetaTuneTabPFNClassifier":
        super().fit(X, y, "density_filter", MyTunedTabPFNClassifier, TuningParams.TABPFN_C0, None)
        return self



class MetaEnsembleTabPFNClassifier(MetaEnsembleBaseEstimator):
    def fit(self, X: XType, y:YType) -> "MetaEnsembleTabPFNClassifier":
        super().fit(X, y, "density_filter", MyEnsembledTabPFNClassifier, TuningParams.TABPFN_C0, None)
        return self