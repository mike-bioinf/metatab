from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from finetabpfn import AesFineTunedTabPFNClassifier
from scipy.stats import randint, loguniform



Classifier: TypeAlias = Union[
    RandomForestClassifier,
    XGBClassifier,
    TabPFNClassifier, 
    AutoTabPFNClassifier, 
    AesFineTunedTabPFNClassifier
]


BoostedClassifier: TypeAlias = Union[
    XGBClassifier
]


SKLEARN_RANDOM_SEARCH_FIXED_PARAMS = {
    "n_iter": 100,
    "scoring": "neg_log_loss",
    "n_jobs": 1,
    "refit": True
}


RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS = {
    "grow_policy": ["depthwise"],
    "tree_method": ["exact"],
    "max_depth": randint(low=2, high=6),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": loguniform(a=0.001, b=5),
    "reg_alpha": loguniform(a=0.001, b=5),
    "gamma": loguniform(a=0.001, b=5),
    "min_child_weight": loguniform(a=0.001, b=5),
    "subsample": [0.8, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


ES_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10000,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "verbose_eval": False,
    "random_state": 0,
    "n_jobs": -1,
    "verbosity": 0
}


RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 700,
    "random_state": 0,
    "n_jobs": -1,
    "verbosity": 0
}
