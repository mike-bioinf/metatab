from __future__ import annotations

import re
import json
import importlib
from typing import Literal, TYPE_CHECKING
from pathlib import Path
from dataclasses import asdict
from metatab.metatab_utils.general import enlist
from metatab.preprocessing.types import PreprocessingStrategy
from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
from metatab.estimators.utils.types import TunableEstimatorType
from metatab.estimators.utils.general import check_meta_tuning_options, check_validation_set_options

from metatab.estimators.utils.constants import (
    NON_EARLY_STOPPED_ESTIMATORS,
    NON_EARLY_STOPPED_CPU_ESTIMATORS,
    NON_EARLY_STOPPED_GPU_ESTIMATORS
)

from metatab.estimators.core.meta_ens_base_estimator import BaseEnsembleEstimator

from metatab.metalearning.utils import (
    BestMetaStrategyParams, 
    RandomFromBestMetaStrategyParams, 
    UniformFromBestMetaStrategyParams,
    RandomUniformFromBestMetaStrategyParams
)

if TYPE_CHECKING:
    from metatab.estimators.estimators import EnsembledEstimator




class CollectionEnsembleEstimators:
    '''
    Class that collect the ensembled estimators to further ensemble.
    Provides capabilities to dump and load json representations,
    and to create pre-defined collections.

    Parameters:
        ensemble_estimators (EnsembledEstimator | list[EnsembledEstimator]) 
        Collection of ensemble estimators to ensemble.
        The estimators must be NOT fitted
    '''
    def __init__(self, ensemble_estimators: EnsembledEstimator | list[EnsembledEstimator]):
        self.ensemble_estimators = enlist(ensemble_estimators)
        self._check_confs()


    def _check_confs(self) -> None:
        for ens in self.ensemble_estimators:
            if not isinstance(ens, BaseEnsembleEstimator):
                raise ValueError("All estimators must be ensemble estimators.")
        
        ens_names = [ens.name for ens in self.ensemble_estimators]

        if len(ens_names) != len(set(ens_names)):
            raise ValueError((
                "Found duplicate ensemble names."
                " The names are used as anchor to the saving locations so cannot be duplicated." 
            ))
        

    def dump_json(self, file: str | Path) -> None:
        '''Dump the CollectionEnsembleEstimators into a json file'''
        json_dict = {}
        for ens in self.ensemble_estimators:
            init_conf = ens._get_init_configuration()
            if init_conf["params"].get("meta_strategy_params"):
                init_conf["params"]["meta_strategy_params"] = asdict(init_conf["params"]["meta_strategy_params"])
            json_dict[ens.name] = init_conf

        with open(file, "w") as f:
            json.dump(json_dict, f, indent=4)


    @classmethod
    def load_json(cls, file: str | Path) -> "CollectionEnsembleEstimators":
        '''Load from the json file the CollectionEnsembleEstimators object'''
        with open(file, "r") as f:
            collection: dict = json.load(f)
        
        instances = []
        for init_conf in collection.values():
            ###REFACTOR: this is weak to internal changes --> find better solutions --> maybe one based on a model registry
            module = importlib.import_module(init_conf["__module__"])
            ens_cls = getattr(module, init_conf["__class__"])
            instances.append(ens_cls(**cls._refine_params(init_conf["params"])))
        
        return cls(instances)
    

    @staticmethod
    def _refine_params(init_conf) -> dict:
        '''Takes care of the meta_strategy_params parameter'''
        meta_strategy_params = init_conf.get("meta_strategy_params", None)

        if meta_strategy_params is None:
            return init_conf
        else:
            meta_strategy = init_conf["meta_strategy"]

            if meta_strategy == "best":
                init_conf["meta_strategy_params"] = BestMetaStrategyParams(**meta_strategy_params)
            elif meta_strategy == "random_from_best":
                init_conf["meta_strategy_params"] = RandomFromBestMetaStrategyParams(**meta_strategy_params)
            elif meta_strategy == "uniform_from_best":
                init_conf["meta_strategy_params"] = UniformFromBestMetaStrategyParams(**meta_strategy_params)
            elif meta_strategy == "random_uniform_from_best":
                init_conf["meta_strategy_params"] = RandomUniformFromBestMetaStrategyParams(**meta_strategy_params)
            else:
                raise ValueError("`meta_strategy` parameter not recognized.")
        
        return init_conf



    ###REFACTOR: broken now to update
    # @classmethod
    # def create_predefined_collection(cls, wildcard: str) -> "CollectionEnsembleEstimators":
    #     '''
    #     Create a predefined collection of user ensemble configurations from a wildcard string.
    #     The wildcard must follow the pattern: (all|cpu|gpu)_(meta|random)_{n_members}

    #     where:
    #     - the first component selects the estimators.
    #     - the second component selects the ensemble algorithm.
    #     - n_members is the number of ensemble members.

    #     Default settings are used for preprocessing, tuning space, and early stopping.

    #     Parameters:
    #         wildcard (str): Wildcard string defining the collection.

    #     Returns:
    #         CollectionEnsembleEstimators
    #     '''        
    #     if not re.match(r'^(all|cpu|gpu)_(meta|random)_\d+$', wildcard):
    #         raise ValueError(
    #             "The wildcard should adere to the pattern (all|cpu|gpu)_(meta|random)_{n_members}."
    #         )
    #     estimators, algo, n_members = wildcard.split("_")
    #     return cls._create_collection(estimators, algo, int(n_members))


    # @classmethod
    # def _create_collection(
    #     cls,
    #     estimators: Literal["all", "cpu", "gpu"],
    #     ensemble_algo: Literal["random", "meta"],
    #     n_members: int
    # ) -> "CollectionEnsembleEstimators":
    #     if estimators == "all":
    #         target_estimators = NON_EARLY_STOPPED_ESTIMATORS + ["realmlp", "tabm"]  ## TODO: adjust this 
    #     elif estimators == "cpu":
    #         target_estimators = NON_EARLY_STOPPED_CPU_ESTIMATORS
    #     elif estimators == "gpu":
    #         target_estimators = NON_EARLY_STOPPED_GPU_ESTIMATORS + ["realmlp", "tabm"] ## TODO: adjust this

    #     collection = []
    #     for i, estimator in enumerate(target_estimators):
    #         collection.append(
    #             UserEnsembleConfiguration(
    #                 name="ens_" + f"{i}",
    #                 algo=ensemble_algo,
    #                 n_members=n_members,
    #                 estimator=estimator,
    #                 preprocessing="estimator_default",
    #                 tune_space="default",
    #                 early_stop_on_validation_set=estimator not in NON_EARLY_STOPPED_ESTIMATORS
    #             )
    #         )

    #     return cls(collection)