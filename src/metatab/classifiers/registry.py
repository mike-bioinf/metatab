from typing import Union, TypeAlias
from metatab.utils.types import EstimatorType 
from metatab.classifiers.extra_trees import ExtraTreeSpec
from metatab.classifiers.random_forest import RandomForestSpec
from metatab.classifiers.catboost import CatBoostSpec, EsCatBoostSpec
from metatab.classifiers.lgbm import LGBMSpec, EsLGBMSpec
from metatab.classifiers.xgboost import XGBSpec, EsXGBSpec
from metatab.classifiers.tabpfn import TabPFNSpec
from metatab.classifiers.tabm import TabMSpec
from metatab.classifiers.realmlp import RealMLPSpec


ClassifierSpec: TypeAlias = Union[
    ExtraTreeSpec,
    RandomForestSpec,
    CatBoostSpec, 
    EsCatBoostSpec,
    XGBSpec, 
    EsXGBSpec,
    LGBMSpec,
    EsLGBMSpec,
    TabPFNSpec,
    TabMSpec,
    RealMLPSpec
]


CLASSIFIER_SPECS_REGISTRY = {
    "extra_trees": ExtraTreeSpec,
    "random_forest": RandomForestSpec,
    "catboost": CatBoostSpec,
    "es_catboost": EsCatBoostSpec,
    "xgb": XGBSpec,
    "es_xgb": EsXGBSpec,
    "lgbm": LGBMSpec,
    "es_lgbm": EsLGBMSpec,
    "tabpfn": TabPFNSpec,
    "tabm": TabMSpec,
    "realmlp": RealMLPSpec
}


def get_classifier_specs_from_registry(type_classifier: EstimatorType) -> ClassifierSpec:
    '''
    Retrieve the ClassifierSpec class of the input classifier.
    Raise an error when `type_classifier` is not present in the registry.
    '''
    if type_classifier not in CLASSIFIER_SPECS_REGISTRY.keys():
        raise KeyError(
            f"The following classifier type is not found in classifiers registry: {type_classifier}."
        )
    return CLASSIFIER_SPECS_REGISTRY[type_classifier]