from typing import TypeAlias, Union

from estimators import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomizedXGBClassifier,
    MyRandomizedESXGBClassifier,
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyTabPFNClassifier
)


Estimator: TypeAlias = Union[
    MyXGBClassifier,
    MyESXGBClassifier,
    MyRandomizedXGBClassifier,
    MyRandomizedESXGBClassifier,
    MyRandomForestClassifier,
    MyRandomizedRandomForestClassifier,
    MyTabPFNClassifier
]