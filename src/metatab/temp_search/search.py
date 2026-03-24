from __future__ import annotations

import sys
import logging
import warnings
import time
import optuna
import numpy as np
from typing import Literal, TYPE_CHECKING
from metatab.utils.general import enlist
from metatab.utils.logging import create_logger
from metatab.temp_search.configuration import PipelineConfiguration
from metatab.temp_search.cross_validator import CrossValidator
from metatab.utils.exceptions import PipelineFitError

if TYPE_CHECKING:
    from metatab.classifiers.registry import ClassifierSpec
    from metatab.preprocessing.types import PreprocessingStrategy



## REFACTOR: complete here
class MetaSearch:
    '''
    Search done via the metaframework.
    '''
    pass



class SearchWithFixedConfiguration:
    '''
    Select the best configurations from a pre-defined list.
    The configurations are evaluated by the evaluator, and the one
    optimizing the score accoring to the chosen direction is selected. 
    '''
    def __init__(
        self,
        confs: list[PipelineConfiguration],
        evaluator: CrossValidator,
        direction_optimization: Literal["minimize", "maximize"],
        log: int,
        time_limit: int
    ):
        self.confs=confs
        self.evaluator=evaluator
        self.direction_optimization=direction_optimization
        self.log=log
        self.time_limit=time_limit

    def get_configurations(self) -> list[PipelineConfiguration]:
        # return the single configuration which is the best by definition
        if len(self.confs) == 1:
            return self.confs
        
        start_time = time.time()

        formatter = logging.Formatter(
            fmt="[%(levelname).1s %(asctime)s,%(msecs)03d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        meta_logger = create_logger("meta_logger", sys.stdout, formatter)
        meta_logger.info("Starting optimation study with fixed configurations.")

        losses = []
        best_conf_index = 0

        for i, conf in enumerate(self.confs):
            if (time.time() - start_time) > self.time_limit:
                break
            ## REFACTOR: complete here
            ## build pipeline from conf
            ## evaluate pipeline
            ## store loss 
            ## evaluate best
            ## log

        if not losses:
            raise ValueError("No trial has been evaluated. Increase the time limit.")
        
        if np.isnan(np.array(losses)).all():
            raise ValueError("All trials have failed.")
        
        return self.confs[best_conf_index]



class OptunaSearch:
    '''
    Search object that uses optuna to perform an optimized search.
    Returns the top configuration resulting from the optimization.
    Currently does not support a multi-classifier search.
    '''
    def __init__(
        self,
        classifier_spec: ClassifierSpec,
        preprocessing: PreprocessingStrategy,
        optuna_sampler: Literal["random", "tpe"],
        evaluator: CrossValidator,
        n_trials: int,
        direction_optimization: Literal["minimize", "maximize"],
        time_limit: int,
        seed: int,
        log: int,
        raise_error_during_search: bool
    ):
        self.classifier_spec=classifier_spec
        self.preprocessing = preprocessing
        self.optuna_sampler = optuna_sampler
        self.evaluator=evaluator
        self.n_trials=n_trials
        self.direction_optimization=direction_optimization
        self.time_limit=time_limit
        self.seed=seed
        self.log=log
        self.raise_error_during_search=raise_error_during_search


    def get_configurations(self) -> list[PipelineConfiguration]:
        optuna.logging.set_verbosity(self.log)

        if self.optuna_sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        else:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=20, # number of random init points, we double the default of 10
                seed=self.seed,
            )
        
        def preprocessing_sampler(trial: optuna.Trial) -> str:
            return trial.suggest_categorical(
                name="__classifier_preprocessing", 
                choices=enlist(self.preprocessing)
            )
        
        if self.n_trials == 1:
            # mock objective
            def objective(trial):
                _ = preprocessing_sampler(trial)
                _ = self.classifier_spec.hps_sampler_function(trial)
                return -1
        else:
            def objective(trial):
                prep = preprocessing_sampler(trial)
                hps = self.classifier_spec.hps_sampler_function(trial)
                loss = safe_evaluator_score(self.evaluator, prep, hps, self.classifier_spec, add_fixed_hps=True)
                return loss
            
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                category=UserWarning, 
                message="Choices for a categorical distribution should be.*"
            )
            study = optuna.create_study(sampler=sampler, direction=self.direction_optimization)
            study.optimize(func=objective, n_trials=self.n_trials, timeout=self.time_limit)

        if not any([t.state == optuna.trial.TrialState.COMPLETE for t in study.trials]):
            raise ValueError("All trials have failed.")
        
        prep = study.best_trial.params.pop("__classifier_preprocessing")
        # this resolve conditional logic returning a classifier-compatible hp dict
        hps = self.classifier_spec.hps_sampler_function(study.best_trial.params)
        best_configuration = PipelineConfiguration(prep, hps, self.classifier_spec)
        # we return a list to comply to other search objects
        return [best_configuration]



class UnoptimizedRandomSearch:
    '''
    Search object that does not perform optimization.
    It samples randomly and returns the sampled configurations.
    Currently does not support a multi-classifier search.
    '''
    def __init__(
        self,
        classifier_spec: ClassifierSpec,
        preprocessing: PreprocessingStrategy,
        n_confs: int,
        seed: int
    ) -> "UnoptimizedRandomSearch":
        self.classifier_spec = classifier_spec
        self.preprocessing = preprocessing
        self.seed = seed
        self.n_confs = n_confs

    def get_configurations(self) -> list[PipelineConfiguration]:
        '''
        Get `n_confs` configurations and returns them.
        '''
        optuna.logging.set_verbosity(optuna.logging.WARNING) # disable logs
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(self.seed))
        preprocessings = enlist(self.preprocessing)
        
        def preprocessing_sampler(trial: optuna.Trial) -> str:
            return trial.suggest_categorical(
                name="__classifier_preprocessing", 
                choices=preprocessings
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                category=UserWarning, 
                message="Choices for a categorical distribution should be.*"
            )
            
            def mock_objective(trial): 
                # we trigger the trial sampling
                preprocessing_sampler(trial)
                self.classifier_spec.hps_sampler_function(trial)
                return 0
            
            study.optimize(mock_objective, n_trials=self.n_confs)
        
        configurations = []
        for t in study.trials:
            prep = t.params.pop("__classifier_preprocessing")
            configurations.append(PipelineConfiguration(prep, t.params, self.classifier_spec))

        return configurations
    


def safe_evaluator_score(
    evaluator: CrossValidator,
    raise_error_during_search: bool, 
    preprocessing: PreprocessingStrategy, 
    hps: dict, 
    classifier_spec: ClassifierSpec,
    add_fixed_hps: bool
) -> float:
    '''
    Utility that controls the error management in the search.
    In brief we always do NOT tollerate errors different from "PipelineFitError" 
    since related to our implemenetation and not the pipeline fitting process itself.
    When a fit error is raised then the management depends on the "raise_error_during_search" flag.
    When tollerated a nan is returned as score.
    '''
    try:
        score = evaluator.score(preprocessing, hps, classifier_spec, add_fixed_hps)
        return score
    except PipelineFitError:
        if raise_error_during_search:
            raise
        return np.nan
    except Exception:
        raise
