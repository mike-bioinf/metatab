from dataclasses import dataclass, asdict
from metatab.classifiers.registry import ClassifierSpec


@dataclass
class PipelineConfiguration:
    preprocessing: str
    hps: dict
    classifier_spec: ClassifierSpec

    def asdict(self) -> dict:
        '''Return asdict(self)'''
        return asdict(self)
