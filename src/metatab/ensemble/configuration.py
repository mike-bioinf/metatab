import re
import json
from typing import Literal
from pathlib import Path
from dataclasses import asdict
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator
from metatab.metatab_utils.general import enlist
from metatab.metatab_utils.device import check_device_estimator_combination
from metatab.preprocessing.types import PreprocessingStrategy
from metatab.metalearning.types import MetaStrategy, MetaStrategyParams
from metatab.estimators.utils.types import TunableEstimatorType
from metatab.estimators.utils.general import check_meta_tuning_options, check_validation_set_options

from metatab.estimators.utils.constants import (
    NON_EARLY_STOPPED_ESTIMATORS,
    NON_EARLY_STOPPED_CPU_ESTIMATORS,
    NON_EARLY_STOPPED_GPU_ESTIMATORS
)

from metatab.metalearning.utils import (
    BestMetaStrategyParams, 
    RandomFromBestMetaStrategyParams, 
    UniformFromBestMetaStrategyParams,
    RandomUniformFromBestMetaStrategyParams,
    check_meta_strategy_params
)



class UserEnsembleConfiguration(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    algo: Literal["random", "meta"]
    n_members: int
    estimator: TunableEstimatorType
    preprocessing: PreprocessingStrategy
    tune_space: str
    early_stop_on_validation_set: bool
    early_stop_rounds: int = 100
    validation_set_size: float = 0.3
    meta_surrogate_model: None | str | Path = None
    meta_strategy: MetaStrategy = "random_uniform_from_best"
    meta_strategy_params: None | MetaStrategyParams = None
    meta_seed: int = 42
    seed: int = 0
    device: Literal["cpu", "cuda", "auto"] = "auto"


    @model_validator(mode="before")
    @classmethod
    def adjust_meta_strategy_params_from_serialized_data(cls, data):
        # in before mode we have only data specified in input without defaults
        # therefore here we use "get" with the default value for meta_strategy 
        meta_strategy = data.get("meta_strategy", "random_uniform_from_best")
        meta_strategy_params = data.get("meta_strategy_params", None)

        # we target only the deserialization case
        if isinstance(
            meta_strategy_params, 
            (
                BestMetaStrategyParams, 
                RandomFromBestMetaStrategyParams, 
                UniformFromBestMetaStrategyParams,
                RandomUniformFromBestMetaStrategyParams
            )
        ):
            return data

        if meta_strategy_params is None:
            return data

        if meta_strategy == "best":
            data["meta_strategy_params"] = BestMetaStrategyParams(**meta_strategy_params)
        elif meta_strategy == "random_from_best":
            data["meta_strategy_params"] = RandomFromBestMetaStrategyParams(**meta_strategy_params)
        elif meta_strategy == "uniform_from_best":
            data["meta_strategy_params"] = UniformFromBestMetaStrategyParams(**meta_strategy_params)
        else:
            data["meta_strategy_params"] = RandomUniformFromBestMetaStrategyParams(**meta_strategy_params)

        return data
    

    @model_validator(mode="after")
    def general_check_after_validation(self) -> "UserEnsembleConfiguration":
        check_validation_set_options(self.estimator, self.early_stop_on_validation_set, self.early_stop_rounds, self.validation_set_size)
        check_device_estimator_combination(self.device, self.estimator)
        check_meta_strategy_params(self.meta_strategy, self.meta_strategy_params, safe_none_params=True)
        check_meta_tuning_options(self.estimator, self.preprocessing, self.tune_space)
        return self


    @field_serializer("meta_strategy_params", when_used="json")
    def serialize(self, value: MetaStrategyParams):
        return None if value is None else asdict(value)
    



class CollectionUserEnsembleConfiguration:
    '''
    Parameters:
        configurations (UserEnsembleConfiguration | list[UserEnsembleConfiguration]): 
            Single or list of UserEnsembleConfiguration objects.
    '''
    def __init__(self, configurations: UserEnsembleConfiguration | list[UserEnsembleConfiguration]):
        self.configurations: list[UserEnsembleConfiguration] = enlist(configurations)
        self._check_confs()


    def _check_confs(self) -> None:
        for conf in self.configurations:
            if not isinstance(conf, UserEnsembleConfiguration):
                raise ValueError(
                    "'configurations' must be a UserEnsembleConfiguration object or a list of them."
                )
        
        conf_names = [conf.name for conf in self.configurations]
        
        if len(conf_names) != len(set(conf_names)):
            raise ValueError(
                "Passed UserEnsembleConfiguration instances with the same name."
            )

    
    def dump_json(self, file: str | Path) -> None:
        '''Dump the CollectionUserEnsembleConfiguration into a json file'''
        with open(file, "w") as f:
            json.dump(
                {conf.name:conf.model_dump(mode="json") for conf in self.configurations},
                f,
                indent=4
            )

    
    @classmethod
    def load_json(cls, file: str | Path) -> "CollectionUserEnsembleConfiguration":
        '''Load from the json file the CollectionUserEnsembleConfiguration object'''
        with open(file, "r") as f:
            data = json.load(f)
        return cls([UserEnsembleConfiguration(**conf_data) for conf_data in data.values()])


    @classmethod
    def create_predefined_collection(cls, wildcard: str) -> "CollectionUserEnsembleConfiguration":
        '''
        Create a predefined collection of user ensemble configurations from a wildcard string.
        The wildcard must follow the pattern: (all|cpu|gpu)_(meta|random)_{n_members}

        where:
        - the first component selects the estimators.
        - the second component selects the ensemble algorithm.
        - n_members is the number of ensemble members.

        Default settings are used for preprocessing, tuning space, and early stopping.

        Parameters:
            wildcard (str): Wildcard string defining the collection.

        Returns:
            CollectionUserEnsembleConfiguration
        '''        
        if not re.match(r'^(all|cpu|gpu)_(meta|random)_\d+$', wildcard):
            raise ValueError(
                "The wildcard should adere to the pattern (all|cpu|gpu)_(meta|random)_{n_members}."
            )
        estimators, algo, n_members = wildcard.split("_")
        return cls._create_collection(estimators, algo, int(n_members))


    @classmethod
    def _create_collection(
        cls,
        estimators: Literal["all", "cpu", "gpu"],
        ensemble_algo: Literal["random", "meta"],
        n_members: int
    ) -> "CollectionUserEnsembleConfiguration":
        if estimators == "all":
            target_estimators = NON_EARLY_STOPPED_ESTIMATORS + ["realmlp", "tabm"]  ## TODO: adjust this 
        elif estimators == "cpu":
            target_estimators = NON_EARLY_STOPPED_CPU_ESTIMATORS
        elif estimators == "gpu":
            target_estimators = NON_EARLY_STOPPED_GPU_ESTIMATORS + ["realmlp", "tabm"] ## TODO: adjust this

        collection = []
        for i, estimator in enumerate(target_estimators):
            collection.append(
                UserEnsembleConfiguration(
                    name="ens" + f"{i}",
                    algo=ensemble_algo,
                    n_members=n_members,
                    estimator=estimator,
                    preprocessing="estimator_default",
                    tune_space="default",
                    early_stop_on_validation_set=estimator not in NON_EARLY_STOPPED_ESTIMATORS
                )
            )

        return cls(collection)