from dataclasses import dataclass
from metatab.classifiers.registry import ClassifierSpec


@dataclass
class PipelineConfiguration:
    preprocessing: str
    hps: dict
    classifier_spec: ClassifierSpec