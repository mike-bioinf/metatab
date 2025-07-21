from typing import TypeAlias, Union
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions_mod.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from finetabpfn import AesFineTunedTabPFNClassifier


Classifier: TypeAlias = Union[
    RandomForestClassifier,
    TabPFNClassifier, 
    AutoTabPFNClassifier, 
    AesFineTunedTabPFNClassifier
]


PRED_DATAFRAME_RESULTS_FIXED_COLUMNS = [
    "dataset", 
    "estimator",
    "splitting_mode",
    "preprocessing",
    "predict_dataset",
    "y_train", 
    "y_test", 
    "pred_proba",
    "repetition", 
    "fold"
]


HPO_DICT_BASE_KEYS = [
    "splitting_mode",
    "preprocessing",
    "repetition",
    "fold"
]