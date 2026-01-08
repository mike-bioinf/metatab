from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

if TYPE_CHECKING:
    from metatab.estimators.estimators.catboost import CatBoostClassifierInterface
    from metatab.estimators.estimators.realmlp import RealMLPClassifier
    from metatab.estimators.estimators.tabm import TabMClassifier


Classifier: TypeAlias = Union[
    RandomForestClassifier,
    ExtraTreesClassifier,
    XGBClassifier,
    "CatBoostClassifierInterface",
    LGBMClassifier,
    TabPFNClassifier,
    "RealMLPClassifier",
    "TabMClassifier"
]


GBDTClassifier: TypeAlias = Union[
    XGBClassifier,
    "CatBoostClassifierInterface",
    LGBMClassifier
]


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