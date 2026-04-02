from __future__ import annotations

import sys
import math
import time
import warnings
import optuna
import numpy as np
from typing import Literal, TYPE_CHECKING
from metatab.utils.general import enlist
from metatab.utils.logging import create_logger
from metatab.utils.exceptions import PipelineFitError
from metatab.search.configuration import PipelineConfiguration
from metatab.search.score import PipelineConfigurationCVScorer

if TYPE_CHECKING:
    from metatab.classifiers.registry import ClassifierSpec
    from metatab.preprocessing import PreprocessingStrategy



class MetaSearch:
    '''
    Search object that levareges metatab meta-framework to propose promising configurations.
    '''
    ## takes classifiers and split n_candidates uniformly?
    # n_candidates_for_classifier = self.n_candidates / len(self.classifier_specs)
    # for classifier_spec in self.classifier_specs:
    #   urs = UnoptimizedRandomSearch()
    #   confs = urs.get_configurations()
    #   build df using confs
    #   surrogate = load_surrogate()
    #   surrogate.predict(df)
    #   select best and return them
    pass



class SearchWithFixedConfiguration:
    '''
    Select the best configuration from a pre-defined list.
    The configurations are scored by the scorer, and the one
    optimizing the score according to the chosen direction is selected. 
    '''
    def __init__(
        self,
        pipe_configurations: list[PipelineConfiguration],
        scorer: PipelineConfigurationCVScorer,
        direction_optimization: Literal["minimize", "maximize"],
        log: int,
        time_limit: int,
        raise_error_during_search: bool
    ):
        self.pipe_configurations=pipe_configurations
        self.scorer=scorer
        self.direction_optimization=direction_optimization
        self.log=log
        self.time_limit=time_limit
        self.raise_error_during_search=raise_error_during_search

    def get_configurations(self) -> list[PipelineConfiguration]:
        # return the single configuration which is the best by definition
        if len(self.pipe_configurations) == 1:
            return self.pipe_configurations
        
        start_time = time.time()
        logger = create_logger("search_fix_conf_logger", sys.stdout, formatter="standard")
        logger.info("Starting optimation study with fixed configurations.")
        
        best_score = -math.inf if self.direction_optimization == "maximize" else math.inf
        scores = []
        best_index = -1 # mock

        for i, pipe_conf in enumerate(self.pipe_configurations):
            if (time.time() - start_time) > self.time_limit: break
            score = safe_score(pipe_conf, self.scorer, self.raise_error_during_search)
            scores.append(score)

            if not np.isnan(score):
                if self.direction_optimization == "maximize":
                    if score > best_score:
                        best_score = score
                        best_index = i
                else:
                    if score < best_score:
                        best_score = score
                        best_index = i
            
            logger.info((
                f"Trial {i} finished with value: {score} and parameters: {pipe_conf.asdict()}."
                f" Best is trial {best_index} with value: {best_score}."
            ))

        if not scores:
            raise ValueError("No trial has been evaluated. Increase the time limit.")
        
        if np.isnan(np.array(scores)).all():
            raise ValueError("All trials have failed.")
        
        return self.pipe_configurations[best_index]



class OptunaSearch:
    '''
    Search object that uses optuna to perform an optimized search.
    Returns the top configuration resulting from the optimization.
    '''
    def __init__(
        self,
        classifier_specs: list[ClassifierSpec],
        preprocessing: PreprocessingStrategy | list[PreprocessingStrategy] | list[list[PreprocessingStrategy]],
        optuna_sampler: Literal["random", "tpe"],
        scorer: PipelineConfigurationCVScorer,
        n_trials: int,
        direction_optimization: Literal["minimize", "maximize"],
        time_limit: int,
        seed: int,
        log: int,
        raise_error_during_search: bool
    ):
        self.classifier_specs=classifier_specs
        self.preprocessing=preprocessing
        self.optuna_sampler=optuna_sampler
        self.scorer=scorer
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
            with warnings.catch_warnings():
                # 'multivariate' and 'group' are experimental interface so they raise warnings
                warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                sampler = optuna.samplers.TPESampler(
                    # number of random init points, we double the default of 10
                    n_startup_trials=20,
                    seed=self.seed,
                    # beneficial since look at hps interconnections
                    multivariate=True,
                    # we partition the search space for each classifier
                    # this is/should inferred by optuna based on parameters co-occurence
                    group=len(self.classifier_specs) > 1,
                    # the multivariate sampling fail in some occasions (i.e, dynamic options),
                    # so it fallback to the independent one raising a warning
                    warn_independent_sampling=False
                )
        
        def preprocessing_sampler(trial: optuna.Trial) -> str:
            return trial.suggest_categorical(
                name="__preprocessing", 
                choices=enlist(self.preprocessing)
            )
        
        def classifier_sampler(trial: optuna.Trial) -> ClassifierSpec:
            return trial.suggest_categorical(
                name="__classifier_spec",
                choices=self.classifier_specs
            )
        
        if self.n_trials == 1:
            # mock objective
            def objective(trial):
                _ = preprocessing_sampler(trial)
                spec = classifier_sampler(trial)
                _ = spec.hps_sampler_function(trial)
                return -1
        else:
            def objective(trial):
                prep = preprocessing_sampler(trial)
                spec =  classifier_sampler(trial)
                hps = spec.hps_sampler_function(trial)
                pipe_conf = PipelineConfiguration(prep, hps, spec)
                score = safe_score(pipe_conf, self.scorer, self.raise_error_during_search)
                return score
            
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
        
        best_prep = study.best_trial.params["__preprocessing"]
        best_spec = study.best_trial.params["__classifier_spec"]
        # this resolve conditional logic returning a classifier-compatible HP dict
        best_hps = best_spec.hps_sampler_function(study.best_trial)
        best_configuration = PipelineConfiguration(best_prep, best_hps, best_spec)
        # we return a list to comply to other search objects
        return [best_configuration]



class UnoptimizedRandomSearch:
    '''
    Search object that does not perform optimization.
    It samples randomly and returns the sampled configurations.
    '''
    def __init__(
        self,
        classifier_specs: list[ClassifierSpec],
        preprocessing: PreprocessingStrategy | list[PreprocessingStrategy] | list[list[PreprocessingStrategy]],
        n_confs: int,
        seed: int
    ) -> "UnoptimizedRandomSearch":
        self.classifier_specs=classifier_specs
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_confs=n_confs

    def get_configurations(self) -> list[PipelineConfiguration]:
        '''
        Get `n_confs` PipelineConfiguration objects and returns them in a list.
        '''
        optuna.logging.set_verbosity(optuna.logging.WARNING) # disable logs
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(self.seed))
        
        def preprocessing_sampler(trial: optuna.Trial) -> PreprocessingStrategy | list[PreprocessingStrategy]:
            return trial.suggest_categorical(
                name="__preprocessing", 
                choices=enlist(self.preprocessing)
            )
        
        def classifier_sampler(trial: optuna.Trial) -> ClassifierSpec:
            return trial.suggest_categorical(
                name="__classifier_spec",
                choices=self.classifier_specs
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                category=UserWarning, 
                message="Choices for a categorical distribution should be.*"
            )
            
            def mock_objective(trial): 
                # we trigger the trial sampling
                _ = preprocessing_sampler(trial)
                sampled_spec = classifier_sampler(trial)
                _ = sampled_spec.hps_sampler_function(trial)
                return 0
            
            study.optimize(mock_objective, n_trials=self.n_confs)
        
        pipe_confs = []
        for t in study.trials:
            prep = t.params["__preprocessing"]
            spec = t.params["__classifier_spec"]
            hps = spec.hps_sampler_function(t)
            pipe_confs.append(PipelineConfiguration(prep, hps, spec))

        return pipe_confs
    


def safe_score(
    pipe_configuration: PipelineConfiguration,
    scorer: PipelineConfigurationCVScorer,
    raise_error_during_search: bool, 
) -> float:
    '''
    Utility that controls the error management in the search.
    In brief we always do NOT tollerate errors different from "PipelineFitError" 
    since related to our implemenetation and not the pipeline fitting process itself.
    When a fit error is raised then the management depends on the "raise_error_during_search" flag.
    When tollerated a nan is returned as score.
    '''
    try:
        score = scorer.score(pipe_configuration)
        return score
    except PipelineFitError:
        if raise_error_during_search:
            raise
        return np.nan
    except Exception:
        raise
