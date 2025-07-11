from copy import deepcopy
from numpy.random import RandomState
from typing import Literal
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fit.estimators.constants import Classifier
from fit.preprocessing import DensityFeatureSelector



def float_to_int(rvs):
    def rvs_wrapper(*args, **kwargs):
        return rvs(*args, **kwargs).round().astype(int)
    return rvs_wrapper



def int_loguniform(low, high):
    '''
    Function to create a loguniform scipy object 
    that returns integers via the "rsv" method.
    '''
    lu = loguniform(low, high)
    lu.rvs = float_to_int(lu.rvs)
    return lu



def get_fresh_random_state(random_state: None | int | RandomState) -> RandomState:
    '''
    Get a fresh random state instance.
    If the input is None it generates a new instance seeded with 0.
    If int then it produces the random state using it as seed.
    If RandomState it returns a deepcopy of it.
    '''
    if random_state is None:
        return RandomState(0)
    elif isinstance(random_state, int):
        return RandomState(random_state)
    elif isinstance(random_state, RandomState):
        return deepcopy(random_state)
    else:
        raise ValueError("Unsupported input.")



def add_string_to_params(params_dict: dict, string: str):
    '''
    Utility to add at the beginning of the keys of the dict 
    of parametersa string. This is helpful when using sklearn pipelines.
    Returns a new dict.
    '''
    return {f"{string}{k}":v for k, v in params_dict.items()}



def create_default_pipeline(
    preprocessing:  Literal["base", "density_filter", "pca"],
    density_feature_selector_strategy: Literal["exact", "oversample", "undersample"],
    classifier: Classifier | None = None,
    classifier_params: dict | None = None,
) -> Pipeline:
    '''
    Creates the standard/most-used pipeline configurations for each preprocessing strategy.
    Allows to adapt the dinamic strategy parameter of the DenistyFeatureSelector.
    Allows to use the classifier as final step or not. 
    If used one must specify also the parameters to pass in it.
    '''
    if preprocessing == "base":
        return make_pipeline(
            *add_classifier_head_to_steps(
                (VarianceThreshold(),), 
                classifier, 
                classifier_params
            )
        )
    
    elif preprocessing == "pca":
        return make_pipeline(
                *add_classifier_head_to_steps(
                (
                    VarianceThreshold(), 
                    StandardScaler(), 
                    PCA(svd_solver="full", n_components=0.95)
                ),
                classifier,
                classifier_params
            )
        )
    
    elif preprocessing =="density_filter":
        return make_pipeline(
                *add_classifier_head_to_steps(
                (
                    VarianceThreshold(), 
                    DensityFeatureSelector(n_target_cols=500, strategy=density_feature_selector_strategy)
                ),
                classifier,
                classifier_params
            )
        )
    
    else:
        raise ValueError("Unsupported preprocessing.")



def add_classifier_head_to_steps(
    steps: tuple,
    classifier: Classifier | None, 
    classifier_params: dict | None
) -> tuple:
    '''
    Add the classifier to the steps if not None.
    If it is None return steps.
    '''
    if classifier is not None:
        steps = [step for step in steps]
        steps.append(classifier(**classifier_params))
        steps = tuple(steps)
    return steps