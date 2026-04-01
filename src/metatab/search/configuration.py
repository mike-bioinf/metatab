from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, asdict
from metatab.classifiers.registry import ClassifierSpec

if TYPE_CHECKING:
    from metatab.preprocessing import PreprocessingStrategy


@dataclass
class PipelineConfiguration:
    preprocessing: PreprocessingStrategy | list[PreprocessingStrategy]
    hps: dict
    classifier_spec: ClassifierSpec

    def asdict(self) -> dict:
        '''Return asdict(self)'''
        return asdict(self)
