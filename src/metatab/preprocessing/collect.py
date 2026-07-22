from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from metatab.preprocessing.utils import resolve_preprocessing_info

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from metatab.preprocessing import DensityFeatureSelector  
    from metatab.preprocessing.types import PreprocessingStrategy



def collect_fit_preprocessing_info(
    pipe: Pipeline,
    preprocessing: PreprocessingStrategy,
    return_on_no_preprocessing: Literal["empty_dict", "error"] = "empty_dict",
    wrap_into_list: bool = True
) -> dict:
    '''
    Returns the preprocessing info from a fitted pipeline.

    Parameters:
        pipe (Pipeline): Fitted pipeline.

        preprocessing (PreprocessingStrategy): 
            Type of preprocessing used for the pipe object.
        
        return_on_no_preprocessing (Literal["empty_dict", "error"]):
            If no preprocessing is done, i.e the pipeline in input is shallow, then:
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
    if len(pipe) == 1:
        if return_on_no_preprocessing == "empty_dict":
            return {}
        elif return_on_no_preprocessing == "error":
            raise ValueError("Shallow pipeline in input. No preprocessing is done.")
        else:
            raise ValueError("Unsupported value for 'return_on_classifier' parameter.")
    
    preprocessing = resolve_preprocessing_info(preprocessing)

    # from here we deal with a deep pipeline
    if preprocessing == "pca":
        return _collect_from_pca_preprocessing(pipe, wrap_into_list)
    elif preprocessing == "density_filter":
        return _collect_from_density_preprocessing(pipe)
    elif preprocessing in ["base", "no"]:
        return {}
    else:
        raise ValueError("Unrecognized preprocessing.")


def _collect_from_pca_preprocessing(pipe: Pipeline, wrap_into_list: bool) -> dict:
    pca: PCA = pipe.named_steps["pca"]
    return {
        "n_pca_components": pca.n_components_,
        "explained_variance_ratio": [pca.explained_variance_ratio_] if wrap_into_list else pca.explained_variance_ratio_,
        "total_explained_variance_ratio": pca.explained_variance_ratio_.sum()
    }


def _collect_from_density_preprocessing(pipe: Pipeline) -> dict:
    density_selector: DensityFeatureSelector = pipe.named_steps["densityfeatureselector"]
    return {
        "density_selection_strategy": density_selector.strategy_,
        "n_target_features": density_selector.n_target_features_,
        "minimum_selected_density_score": density_selector.minimum_density_score_
    }