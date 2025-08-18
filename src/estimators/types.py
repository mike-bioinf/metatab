from typing import TypeAlias, Union

from estimators import (
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyTabPFNClassifier
)


Estimator: TypeAlias = Union[
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyTabPFNClassifier
]