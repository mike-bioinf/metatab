from __future__ import annotations

from typing import TYPE_CHECKING
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from metatab.preprocessing.density_selector import DensityFeatureSelector

if TYPE_CHECKING:
    from metatab.estimators.utils.types import Classifier
    from metatab.preprocessing.types import PreprocessingStrategy



def create_classification_pipeline(
    classifier_cls: Classifier,
    preprocessing: PreprocessingStrategy,
) -> Pipeline:
    '''
    Creates the classification pipeline.
   
    Parameters:
        classifier_cls (Classifier):
            Class of the classifier to add as pipeline head. 
        
        preprocessing (PreprocessingStrategy): 
            Preprocessing strategy to follow. Supported values:
            - "no": No preprocessing, returns bare classifier
            - "base": VarianceThreshold only
            - "pca": VarianceThreshold + StandardScaler + PCA
            - "density_filter": VarianceThreshold + DensityFeatureSelector    

    Returns:
        Pipeline: The pipeline object.
    '''
    ## REVIEW: 
    # complete with new preprocessing. 
    # since we return always a pipeline of prep steps + classifiers we can do this in one line
    if preprocessing == "no":
        return make_pipeline(classifier_cls())
    elif preprocessing == "base":
        return make_pipeline(VarianceThreshold(), classifier_cls())
    elif preprocessing == "pca":
        return make_pipeline(VarianceThreshold(), StandardScaler(), PCA(svd_solver="full", n_components=0.95), classifier_cls())
    elif preprocessing == "density_filter":
        return make_pipeline(VarianceThreshold(), DensityFeatureSelector(n_target_cols=500), classifier_cls())
    else:
        raise ValueError("Unsupported preprocessing.")
