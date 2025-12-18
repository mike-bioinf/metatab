from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from metatab.preprocessing.density_selector import DensityFeatureSelector

if TYPE_CHECKING:
    from metatab.estimators.utils.types import Classifier, EstimatorType
    from metatab.preprocessing.types import PreprocessingStrategy



def create_classifier_pipeline(
    *,
    preprocessing: PreprocessingStrategy,
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None,
    type_estimator: EstimatorType | None = None
) -> Classifier | Pipeline:
    '''
    Creates the classifier pipeline for each preprocessing strategy.
    This function supports flexible configurations: it can return a pipeline with 
    preprocessing + classifier, preprocessing only, or just the classifier.

    Parameters:
        preprocessing (PREPROCESSING): 
            Preprocessing strategy to follow. Supported values:
            - "no": No preprocessing, returns bare classifier
            - "base": VarianceThreshold only
            - "pca": VarianceThreshold + StandardScaler + PCA
            - "density_filter": VarianceThreshold + DensityFeatureSelector
            - "estimator_default": Automatically selects preprocessing based on type_estimator
        
        density_feature_selector_strategy (Literal["exact", "oversample", "undersample"]):
            Type of density selection to follow when `preprocessing` equal "density_filter".
            
        classifier (Classifier | None, optional):
            Class of the classifier to add as pipeline head. 
            If None the sole preprocessing pipeline is returned.
        
        classifier_params (dict | None, optional):
            Paramaters used to instatiate the classifier.
            Can be None when `classifier` is None, otherwise must be specified.

        type_estimator (EstimatorType | None, optional):
            String estimator type required when `preprocessing` is "estimator_default". 

    Returns:
        Classifier|Pipeline:
        The pipeline or classifier object.
    '''
    if preprocessing == "no" and classifier is None:
        raise ValueError(
            "Classifier must be specified with 'no' preprocessing."
        )
    
    if classifier is not None and classifier_params is None:
        raise ValueError(
            "'classifier_params' mus be specified when 'classifier' is specified."
        )
    
    if classifier_params is not None and classifier is None:
        raise ValueError(
            "''classifier_params' is specified but 'classifier' is None."
        )
    
    if preprocessing == "estimator_default":
        if type_estimator is None:
            raise ValueError(
                "'type_estimator' must be specified with 'estimator_default' preprocessing"
            )
        preprocessing = get_estimator_default_preprocessing(type_estimator)

    if preprocessing == "no":
        return classifier(**classifier_params)
    elif preprocessing == "base":
        return _create_base_pipeline(classifier, classifier_params)
    elif preprocessing == "pca":
        return _create_pca_pipeline(classifier, classifier_params)
    elif preprocessing == "density_filter":
        return _create_density_filter_pipeline(
            density_feature_selector_strategy,
            classifier,
            classifier_params
        )
    else:
        raise ValueError("Unsupported preprocessing.")


# this is used also in helper programs
def get_estimator_default_preprocessing(type_estimator: EstimatorType):
    if type_estimator == "tabpfn":
        return "density_filter"
    else:
        return "base"


def _create_base_pipeline(
    classifier: Classifier | None = None,
    classifier_params: dict | None = None
) -> Pipeline:
    return make_pipeline(
        *_add_classifier_head_to_steps(
            (VarianceThreshold(),), 
            classifier, 
            classifier_params
        )
    )


def _create_pca_pipeline(
    classifier: Classifier | None = None,
    classifier_params: dict | None = None  
) -> Pipeline:
    return make_pipeline(
        *_add_classifier_head_to_steps(
            (
                VarianceThreshold(), 
                StandardScaler(), 
                PCA(svd_solver="full", n_components=0.95)
            ),
            classifier,
            classifier_params
        )
    )


def _create_density_filter_pipeline(
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None   
) -> Pipeline:
    return make_pipeline(
        *_add_classifier_head_to_steps(
            (
                VarianceThreshold(), 
                DensityFeatureSelector(
                    n_target_cols=500, 
                    strategy=density_feature_selector_strategy,
                    on_empty="select_all"
                )
            ),
            classifier,
            classifier_params
        )
    )


def _add_classifier_head_to_steps(
    steps: tuple,
    classifier: Classifier | None, 
    classifier_params: dict | None
) -> tuple:
    '''
    Add the classifier to steps if not None otherwise return the original steps.
    '''
    if classifier is not None:
        steps = list(steps)
        steps.append(classifier(**classifier_params))
    return tuple(steps)