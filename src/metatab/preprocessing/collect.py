from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from sklearn.pipeline import Pipeline
from metatab.preprocessing.utils import resolve_preprocessing_info

if TYPE_CHECKING:
    from sklearn.decomposition import PCA
    from metatab.estimators.utils.types import Classifier
    from metatab.preprocessing import DensityFeatureSelector  
    from metatab.preprocessing.types import PreprocessingStrategy



def collect_fit_preprocessing_info(
    clf_or_pipe: Classifier | Pipeline,
    preprocessing: PreprocessingStrategy,
    return_on_classifier: Literal["empty_dict", "error"] = "empty_dict",
    wrap_into_list: bool = True
) -> dict:
    '''
    Returns the preprocessing info from a fitted classifier or pipeline.

    Parameters:
        clf_or_pipe (Classifier | Pipeline): 
            Fitted classifier or pipeline.

        preprocessing (PreprocessingStrategy): 
            Type of preprocessing used for clf_or_pipe object.
        
        return_on_classifier (Literal["empty_dict", "error"]):
            If no preprocessing is used (no pipeline object) then:
            - if "empty_dict", an empty dict is returned
            - if "error", an error is raised
        
        wrap_into_list (bool):
            Whether to wrap object with len > 1 into list, 
            in order to return all length 1 object. 
            This is important when these info have to inserted
            into a prediction dataframe.

    Returns:
        dict: The dictionary with the fit preprocessing info.
    '''
    if not isinstance(clf_or_pipe, Pipeline):
        if return_on_classifier == "empty_dict":
            return {}
        elif return_on_classifier == "error":
            raise ValueError("Classifier in input. No preprocessing is done.")
        else:
            raise ValueError("Unsupported value for 'return_on_classifier' parameter.")
    
    preprocessing = resolve_preprocessing_info(preprocessing)

    # from here we deal with a pipeline
    if preprocessing == "pca":
        return _collect_from_pca_preprocessing(clf_or_pipe, wrap_into_list)
    elif preprocessing == "density_filter":
        return _collect_from_density_preprocessing(clf_or_pipe)
    elif preprocessing in ["base", "no"]:
        return {}
    else:
        raise ValueError("Unrecognized preprocessing.")


def _collect_from_pca_preprocessing(pipeline: Pipeline, wrap_into_list: bool) -> dict:
    pca: PCA = pipeline.named_steps["pca"]
    return {
        "n_pca_components": pca.n_components_,
        "explained_variance_ratio": [pca.explained_variance_ratio_] if wrap_into_list else pca.explained_variance_ratio_,
        "total_explained_variance_ratio": pca.explained_variance_ratio_.sum()
    }


def _collect_from_density_preprocessing(pipeline: Pipeline) -> dict:
    density_selector: DensityFeatureSelector = pipeline.named_steps["densityfeatureselector"]
    return {
        "density_selection_strategy": density_selector.strategy_,
        "n_target_features": density_selector.n_target_features_,
        "minimum_selected_density_score": density_selector.minimum_density_score_
    }