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
    from metatab.estimators.estimators.realmlp import RealMLPClassifier
    from metatab.estimators.estimators.tabm import TabMClassifier


Classifier: TypeAlias = Union[
    RandomForestClassifier,
    ExtraTreesClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier,
    TabPFNClassifier,
    "RealMLPClassifier",
    "TabMClassifier"
]


GBDTClassifier: TypeAlias = Union[
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier
]

##refactor: estimator_type is confusing maybe better classifier something?? 
TunableEstimatorType = Literal[
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


EsEstimatorType: Literal[
    "es_lgbm",
    "es_xgb",
    "es_catboost",
    "realmlp",
    "tabm"
]


GBDTEstimatorType = Literal[
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


TreeEstimatorType = Literal[
    "random_forest",
    "extra_trees",
    "xgb",
    "es_xgb",
    "lgbm",
    "es_lgbm",
    "catboost",
    "es_catboost"
]


EstimatorType = Literal[
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


XType: TypeAlias = Union[
    pd.DataFrame,
    np.ndarray
]

YType: TypeAlias = Union[
    pd.Series,
    np.ndarray
]