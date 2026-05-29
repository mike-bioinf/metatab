## This resolve the sklearn and scipy incompatibility issues in the array API
## This location happen to be imported first before other imports and therefore
## the env variable is set first that sklearn is imported.
## TODO: find a more robust way to enfsure this
import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from metatab.metatab_utils.package_data import get_example_data
from metatab.ensemble.family import FamilyEnsembleEstimator
from metatab.ensemble.configuration import UserEnsembleConfiguration, CollectionUserEnsembleConfiguration
from metatab.metatab_utils.prediction.dataframe import PredictionDataframe

from metatab.metalearning.utils import  (
    BestMetaStrategyParams,
    RandomFromBestMetaStrategyParams,
    UniformFromBestMetaStrategyParams,
    RandomUniformFromBestMetaStrategyParams
)

from metatab.estimators.pycore.estimators import (
    MetaTuneRandomForestClassifier,
    StandardTuneRandomForestClassifier,
    MetaTuneExtraTreesClassifier,
    StandardTuneExtraTreesClassifier,
    MetaTuneTabPFNClassifier,
    StandardTuneTabPFNClassifier,
    MetaTuneRealMLPClassifier,
    StandardTuneRealMLPClassifier,
    MetaTuneTabMClassifier,
    StandardTuneTabMClassifier,
    MetaTuneEsLGBMClassifier,
    StandardTuneEsLGBMClassifier,
    MetaTuneLGBMClassifier,
    StandardTuneLGBMClassifier,
    MetaTuneEsCatBoostClassifier,
    StandardTuneEsCatBoostClassifier,
    MetaTuneCatBoostClassifier,
    StandardTuneCatBoostClassifier,
    MetaTuneEsXGBClassifier,
    StandardTuneEsXGBClassifier,
    MetaTuneXGBClassifier,
    StandardTuneXGBClassifier,
    MetaEnsembleRandomForestClassifier,
    StandardEnsembleRandomForestClassifier,
    MetaEnsembleExtraTreesClassifier,
    StandardEnsembleExtraTreesClassifier,
    MetaEnsembleTabPFNClassifier,
    StandardEnsembleTabPFNClassifier,
    MetaEnsembleRealMLPClassifier,
    StandardEnsembleRealMLPClassifier,
    MetaEnsembleTabMClassifier,
    StandardEnsembleTabMClassifier,
    MetaEnsembleEsLGBMClassifier,
    StandardEnsembleEsLGBMClassifier,
    MetaEnsembleLGBMClassifier,
    StandardEnsembleLGBMClassifier,
    MetaEnsembleEsCatBoostClassifier,
    StandardEnsembleEsCatBoostClassifier,
    MetaEnsembleCatBoostClassifier,
    StandardEnsembleCatBoostClassifier,
    MetaEnsembleEsXGBClassifier,
    StandardEnsembleEsXGBClassifier,
    MetaEnsembleXGBClassifier,
    StandardEnsembleXGBClassifier
)


__all__ = [
    "MetaTuneRandomForestClassifier",
    "StandardTuneRandomForestClassifier",
    "MetaTuneExtraTreesClassifier",
    "StandardTuneExtraTreesClassifier",
    "MetaTuneTabPFNClassifier",
    "StandardTuneTabPFNClassifier",
    "MetaTuneRealMLPClassifier",
    "StandardTuneRealMLPClassifier",
    "MetaTuneTabMClassifier",
    "StandardTuneTabMClassifier",
    "MetaTuneEsLGBMClassifier",
    "StandardTuneEsLGBMClassifier",
    "MetaTuneLGBMClassifier",
    "StandardTuneLGBMClassifier",
    "MetaTuneEsCatBoostClassifier",
    "StandardTuneEsCatBoostClassifier",
    "MetaTuneCatBoostClassifier",
    "StandardTuneCatBoostClassifier",
    "MetaTuneEsXGBClassifier",
    "StandardTuneEsXGBClassifier",
    "MetaTuneXGBClassifier",
    "StandardTuneXGBClassifier",
    "MetaEnsembleRandomForestClassifier",
    "StandardEnsembleRandomForestClassifier",
    "MetaEnsembleExtraTreesClassifier",
    "StandardEnsembleExtraTreesClassifier",
    "MetaEnsembleTabPFNClassifier",
    "StandardEnsembleTabPFNClassifier",
    "MetaEnsembleRealMLPClassifier",
    "StandardEnsembleRealMLPClassifier",
    "MetaEnsembleTabMClassifier",
    "StandardEnsembleTabMClassifier",
    "MetaEnsembleEsLGBMClassifier",
    "StandardEnsembleEsLGBMClassifier",
    "MetaEnsembleLGBMClassifier",
    "StandardEnsembleLGBMClassifier",
    "MetaEnsembleEsCatBoostClassifier",
    "StandardEnsembleEsCatBoostClassifier",
    "MetaEnsembleCatBoostClassifier",
    "StandardEnsembleCatBoostClassifier",
    "MetaEnsembleEsXGBClassifier",
    "StandardEnsembleEsXGBClassifier",
    "MetaEnsembleXGBClassifier",
    "StandardEnsembleXGBClassifier",

    "get_example_data",
    "FamilyEnsembleEstimator",
    "PredictionDataframe",
    "BestMetaStrategyParams",
    "RandomFromBestMetaStrategyParams",
    "UniformFromBestMetaStrategyParams",
    "RandomUniformFromBestMetaStrategyParams",
    "UserEnsembleConfiguration",
    "CollectionUserEnsembleConfiguration"
]