from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, TYPE_CHECKING
from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier

if TYPE_CHECKING:
    from metatab.classifiers.realmlp import RealMLPClassifier
    from metatab.classifiers.tabm import TabMClassifier
    from metatab.classifiers.autogluon import AutoGluonClassifier



Classifier: TypeAlias = Union[
    RandomForestClassifier,
    ExtraTreesClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier,
    TabPFNClassifier,
    "RealMLPClassifier",
    "TabMClassifier",
    "AutoGluonClassifier"
]


TunableClassifierType = Literal[
    "random_forest",
    "extra_trees",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn",
    "realmlp",
    "tabm"
]


DefaultClassifierType = Literal[
    "random_forest",
    "extra_trees",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost",
    "tabpfn",
    "realmlp",
    "tabm",
    "autogluon"
]


XType: TypeAlias = Union[
    pd.DataFrame,
    np.ndarray
]


YType: TypeAlias = Union[
    pd.Series,
    np.ndarray
]