from typing import TypeAlias, Union

from estimators import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomizedXGBClassifier,
    MyESRandomizedXGBClassifier,
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyTabPFNClassifier
)


Estimator: TypeAlias = Union[
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomizedXGBClassifier,
    MyESRandomizedXGBClassifier,
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyTabPFNClassifier
]