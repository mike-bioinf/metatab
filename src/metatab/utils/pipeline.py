from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
from sklearn.pipeline import Pipeline
from metatab.utils.general import ensure_or_create
from metatab.preprocessing import build_preprocessing_pipeline

if TYPE_CHECKING:
    from metatab.utils.types import YType
    from metatab.preprocessing import PreprocessingStrategy
    from metatab.classifiers.registry import ClassifierSpec



def build_pipeline(
    *,
    preprocessing: PreprocessingStrategy | list[PreprocessingStrategy], 
    hps: dict, 
    classifier_spec: ClassifierSpec,
    classifier_seed: int,
    classifier_device: Literal["cpu", "cuda", "auto"],
    classifier_nthreads: int,
    y: YType
) -> Pipeline:
    '''
    Utility that builds a pipeline (preprocessing + classifier).
    The classifier step is named as "classifier" in the resulting pipeline.
    The preprocessing steps are named after the strategies.
    Handles the device "auto" specification.

    Parameters:
        preprocessing (PreprocessingStrategy | list[PreprocessingStrategy]): 
            Preprocessing/s to use.    
        
        hps (dict): Classifier hps.

        classifier_spec (ClassifierSpec): Classifier dataclass.
        
        classifier_seed (int): 
            Integer used to seed the classifier object.
            When the classifier has no a random_state-like paramater it is ignored.

        y (YType): y labels. Needed for callbacks mechanisms.

    Returns:
        Pipeline: The pipeline object.
    '''
    preprocessing_pipeline = build_preprocessing_pipeline(preprocessing)

    if classifier_device == "auto":
        if classifier_spec.main_device == "cuda" and torch.cuda.is_available():
            classifier_device = "cuda"
        else:
            classifier_device = "cpu"
        
    hps = {
        **hps,
        **{
            param: value
            for param, value in [
                (classifier_spec.random_state_parameter, classifier_seed),
                (classifier_spec.n_threads_parameter, classifier_nthreads),
                (classifier_spec.device_parameter, classifier_device),
            ]
            if param is not None
        }
    }
    
    for callback in ensure_or_create(classifier_spec.callbacks_on_params, list):
        hps = callback(hps, y, False)

    classifier = classifier_spec.classifier_class()
    classifier_spec.set_params_function(classifier, hps)
    full_pipeline = Pipeline(preprocessing_pipeline.steps + [("classifier", classifier)])
    return full_pipeline