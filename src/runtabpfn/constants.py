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


PRED_DATAFRAME_ADDITIONAL_COLUMNS = [
    "model",
    "test_dataset",
    "splitting_mode",
    "repetition", 
    "fold", 
    "preprocessing",
    "number_initial_features",
    "number_filtered_features",
    "filtering_threshold",
    "number_pca_components"
]


HPO_DICT_BASE_KEYS = [
    "splitting_mode",
    "preprocessing",
    "repetition",
    "fold"
]