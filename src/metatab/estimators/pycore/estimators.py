from metatab.estimators.estimators.rf import MyTunedRandomForestClassifier, MyEnsembledRandomForestClassifier
from metatab.estimators.estimators.extra_trees import MyTunedExtraTreesClassifier, MyEnsembledExtraTreesClassifier
from metatab.estimators.estimators.tabpfn import MyTunedTabPFNClassifier, MyEnsembledTabPFNClassifier
from metatab.estimators.estimators.tabm import MyTunedTabMClassifier, MyEnsembledTabMClassifier
from metatab.estimators.estimators.realmlp import MyTunedRealMLPClassifier, MyEnsembledRealMLPClassifier

from metatab.estimators.estimators.catboost import (
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier
)

from metatab.estimators.estimators.lgbm import (
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier, 
    MyEnsembledESLGBMClassifier
)

from metatab.estimators.estimators.xgb import (
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier, 
    MyEnsembledESXGBClassifier
)

from metatab.estimators.pycore.fit_interface import (
    FitInterfaceNoES,
    FitInterfaceWithES,
    FitInterfaceWithFullES,
)

from metatab.estimators.pycore.base_ensemble import (
    BaseEnsemble, 
    StandardEnsembleInitializer, 
    MetaEnsembleInitializer
)

from metatab.estimators.pycore.base_tune import (
    StandardTuneInitializer,
    MetaTuneInitializer,
    BaseTune
)



#### Meta tuned classes -------------------------------------------------------------------------------------------------

class MetaTuneRandomForestClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="random_forest"
    myclass=MyTunedRandomForestClassifier
    enforce_meta_algo=True


class MetaTuneExtraTreesClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="extra_trees"
    myclass=MyTunedExtraTreesClassifier
    enforce_meta_algo=True


class MetaTuneTabPFNClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="tabpfn"
    myclass=MyTunedTabPFNClassifier
    enforce_meta_algo=True


class MetaTuneRealMLPClassifier(MetaTuneInitializer, FitInterfaceWithES, BaseTune):
    type_estimator="realmlp"
    myclass=MyTunedRealMLPClassifier
    enforce_meta_algo=True


class MetaTuneTabMClassifier(MetaTuneInitializer, FitInterfaceWithES, BaseTune):
    type_estimator="tabm"
    myclass=MyTunedTabMClassifier
    enforce_meta_algo=True


class MetaTuneEsLGBMClassifier(MetaTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_lgbm"
    myclass=MyTunedESLGBMClassifier
    enforce_meta_algo=True


class MetaTuneLGBMClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="lgbm"
    myclass=MyTunedLGBMClassifier
    enforce_meta_algo=True


class MetaTuneEsCatBoostClassifier(MetaTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_catboost"
    myclass=MyTunedESCatBoostClassifier
    enforce_meta_algo=True


class MetaTuneCatBoostClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="catboost"
    myclass=MyTunedCatBoostClassifier
    enforce_meta_algo=True


class MetaTuneEsXGBClassifier(MetaTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_xgb"
    myclass=MyTunedESXGBClassifier
    enforce_meta_algo=True


class MetaTuneXGBClassifier(MetaTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="xgb"
    myclass=MyTunedXGBClassifier
    enforce_meta_algo=True



### ---- Standard tuned classes ------------------------------------------------------------------------------------------

class StandardTuneRandomForestClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="random_forest"
    myclass=MyTunedRandomForestClassifier
    enforce_meta_algo=False


class StandardTuneExtraTreesClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="extra_trees"
    myclass=MyTunedExtraTreesClassifier
    enforce_meta_algo=False


class StandardTuneTabPFNClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="tabpfn"
    myclass=MyTunedTabPFNClassifier
    enforce_meta_algo=False


class StandardTuneRealMLPClassifier(StandardTuneInitializer, FitInterfaceWithES, BaseTune):
    type_estimator="realmlp"
    myclass=MyTunedRealMLPClassifier
    enforce_meta_algo=False


class StandardTuneTabMClassifier(StandardTuneInitializer, FitInterfaceWithES, BaseTune):
    type_estimator="tabm"
    myclass=MyTunedTabMClassifier
    enforce_meta_algo=False


class StandardTuneEsLGBMClassifier(StandardTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_lgbm"
    myclass=MyTunedESLGBMClassifier
    enforce_meta_algo=False


class StandardTuneLGBMClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="lgbm"
    myclass=MyTunedLGBMClassifier
    enforce_meta_algo=False


class StandardTuneEsCatBoostClassifier(StandardTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_catboost"
    myclass=MyTunedESCatBoostClassifier
    enforce_meta_algo=False


class StandardTuneCatBoostClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="catboost"
    myclass=MyTunedCatBoostClassifier
    enforce_meta_algo=False


class StandardTuneEsXGBClassifier(StandardTuneInitializer, FitInterfaceWithFullES, BaseTune):
    type_estimator="es_xgb"
    myclass=MyTunedESXGBClassifier
    enforce_meta_algo=False


class StandardTuneXGBClassifier(StandardTuneInitializer, FitInterfaceNoES, BaseTune):
    type_estimator="xgb"
    myclass=MyTunedXGBClassifier
    enforce_meta_algo=False




# ----- Meta ensembled classes  -------------------------------------------------------------------------------------------------

class MetaEnsembleRandomForestClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "random_forest"
    myclass = MyEnsembledRandomForestClassifier


class MetaEnsembleExtraTreesClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "extra_trees"
    myclass = MyEnsembledExtraTreesClassifier


class MetaEnsembleTabPFNClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "tabpfn"
    myclass = MyEnsembledTabPFNClassifier


class MetaEnsembleRealMLPClassifier(MetaEnsembleInitializer, FitInterfaceWithES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "realmlp"
    myclass = MyEnsembledRealMLPClassifier


class MetaEnsembleTabMClassifier(MetaEnsembleInitializer, FitInterfaceWithES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "tabm"
    myclass = MyEnsembledTabMClassifier


class MetaEnsembleEsLGBMClassifier(MetaEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "es_lgbm"
    myclass = MyEnsembledESLGBMClassifier


class MetaEnsembleLGBMClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "lgbm"
    myclass = MyEnsembledLGBMClassifier


class MetaEnsembleEsCatBoostClassifier(MetaEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "es_catboost"
    myclass = MyEnsembledESCatBoostClassifier


class MetaEnsembleCatBoostClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "catboost"
    myclass = MyEnsembledCatBoostClassifier


class MetaEnsembleEsXGBClassifier(MetaEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "es_xgb"
    myclass = MyEnsembledESXGBClassifier


class MetaEnsembleXGBClassifier(MetaEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "meta"
    type_estimator = "xgb"
    myclass = MyEnsembledXGBClassifier




# ----- Standard ensembled classes --------------------------------------------------------------------------------------------------

class StandardEnsembleRandomForestClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "random_forest"
    myclass = MyEnsembledRandomForestClassifier


class StandardEnsembleExtraTreesClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "extra_trees"
    myclass = MyEnsembledExtraTreesClassifier


class StandardEnsembleTabPFNClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "tabpfn"
    myclass = MyEnsembledTabPFNClassifier


class StandardEnsembleRealMLPClassifier(StandardEnsembleInitializer, FitInterfaceWithES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "realmlp"
    myclass = MyEnsembledRealMLPClassifier


class StandardEnsembleTabMClassifier(StandardEnsembleInitializer, FitInterfaceWithES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "tabm"
    myclass = MyEnsembledTabMClassifier


class StandardEnsembleEsLGBMClassifier(StandardEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "es_lgbm"
    myclass = MyEnsembledESLGBMClassifier


class StandardEnsembleLGBMClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "lgbm"
    myclass = MyEnsembledLGBMClassifier


class StandardEnsembleEsCatBoostClassifier(StandardEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "es_catboost"
    myclass = MyEnsembledESCatBoostClassifier


class StandardEnsembleCatBoostClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "catboost"
    myclass = MyEnsembledCatBoostClassifier


class StandardEnsembleEsXGBClassifier(StandardEnsembleInitializer, FitInterfaceWithFullES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "es_xgb"
    myclass = MyEnsembledESXGBClassifier


class StandardEnsembleXGBClassifier(StandardEnsembleInitializer, FitInterfaceNoES, BaseEnsemble):
    type_ensemble = "random"
    type_estimator = "xgb"
    myclass = MyEnsembledXGBClassifier
