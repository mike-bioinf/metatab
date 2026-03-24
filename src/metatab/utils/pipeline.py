from __future__ import annotations

from typing import TYPE_CHECKING
from metatab.classifiers.registry import ClassifierSpec

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.preprocessing.types import PreprocessingStrategy



def build_pipeline(
    preprocessing: PreprocessingStrategy, 
    hps: dict, 
    classifier_spec: ClassifierSpec, ## can be smplified with the name of the random_state_parameter ???
    classifier_seed: int
) -> Pipeline:
    '''
    Utility that builds the pipeline using the preprocessing and dynamic hps in input.

    Parameters:
        preprocessing (PreprocessingStrategy): Preprocessing to use.
        
        hps (dict): classifier hps.
        
        classifier_spec (ClassifierSpec): 
            Classifier dataclass of which we are building the pipeline.
        
        classifier_seed (int): 
            integer used to seed the classifier object.
            When the classifier has no a random_state-like paramater it is ignored.

    Returns:
        Pipeline: The pipeline object.
    '''
    pass
    # params_to_fit = {**params, self.clf_random_state_parameter: int(rng.integers(0, 2**32))}
    ### ATTENZIONE AL SET. Qui dobbiamo finalizzare i params.
    # set_params_into_clf(pipe, params_to_fit, set_tabpfn_inference_config=True, finalize_tabpfn_model_path=True)