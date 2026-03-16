from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from metatab.preprocessing.density_selector import DensityFeatureSelector

if TYPE_CHECKING:
    from metatab.utils.types import Classifier
    from metatab.preprocessing.types import ResolvedPreprocessingStrategy



def create_classifier_pipeline(
    *,
    preprocessing: ResolvedPreprocessingStrategy,
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None,
) -> Pipeline:
    '''
    Creates the pipeline for each preprocessing strategy.
    This function supports flexible configurations: it can return a pipeline with 
    preprocessing + classifier, or preprocessing and classifier only.
    The functions always returns a pipeline even when composed by a single estimator.

    Parameters:
        preprocessing (ResolvedPreprocessingStrategy): 
            Preprocessing strategy to follow. Supported values:
            - "no": No preprocessing
            - "base": VarianceThreshold only
            - "pca": VarianceThreshold + StandardScaler + PCA
            - "density_filter": VarianceThreshold + DensityFeatureSelector
        
        density_feature_selector_strategy (Literal["exact", "oversample", "undersample"]):
            Type of density selection to follow when `preprocessing` equal "density_filter".
            
        classifier (Classifier | None, optional):
            Class of the classifier to add as pipeline head. 
            If None the sole preprocessing pipeline is returned.
        
        classifier_params (dict | None, optional):
            Paramaters used to instatiate the classifier.
            Can be None when `classifier` is None, otherwise must be specified.

    Returns:
        Pipeline: The pipeline object.
    '''
    if preprocessing == "no" and classifier is None:
        raise ValueError("Classifier must be specified with 'no' preprocessing.")
    
    if classifier is not None and classifier_params is None:
        raise ValueError("'classifier_params' must be specified when 'classifier' is specified.")
    
    if classifier_params is not None and classifier is None:
        raise ValueError("''classifier_params' is specified but 'classifier' is None.")
    
    if preprocessing == "no":
        return make_pipeline(classifier(**classifier_params))
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


def _create_base_pipeline(
    classifier: Classifier | None = None,
    classifier_params: dict | None = None
) -> Pipeline:
    return make_pipeline(
        *_add_classifier_head_to_steps(
            [VarianceThreshold()], 
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
            [
                VarianceThreshold(), 
                StandardScaler(), 
                PCA(svd_solver="full", n_components=0.95)
            ],
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
            [
                VarianceThreshold(), 
                DensityFeatureSelector(
                    n_target_cols=500, 
                    strategy=density_feature_selector_strategy,
                    on_empty="select_all"
                )
            ],
            classifier,
            classifier_params
        )
    )


def _add_classifier_head_to_steps(
    steps: list,
    classifier: Classifier | None, 
    classifier_params: dict | None
) -> list:
    '''
    Add the classifier to steps if not None otherwise return the original steps.
    '''
    if classifier is not None:
        steps.append(classifier(**classifier_params))
    return steps