import optuna
import warnings
from typing import Callable
from sklearn.utils.validation import check_is_fitted


class WrapperRandomSampler:
    '''
    Wrapper of optuna RandomSampler used to get the list of sampled points.
    '''
    def fit(self, sampler_function: Callable[[optuna.Trial], dict], seed: int) -> "WrapperRandomSampler":
        self.sampler_function = sampler_function
        self.seed = seed
        self.is_fitted_ = True
        return self

    def sample_points(self, n_points: int) -> list[dict]:
        '''
        Get `n_points` from the space.
        Returns the list of sampled points.
        '''
        check_is_fitted(self, "is_fitted_")
        optuna.logging.set_verbosity(optuna.logging.WARNING) # disable logs
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(self.seed))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                category=UserWarning, 
                message="Choices for a categorical distribution should be.*"
            )
            
            def mock_objective(trial): 
                # we trigger the trial sampling
                self.sampler_function(trial)
                return 0
            
            study.optimize(mock_objective, n_trials=n_points)

        return [self.sampler_function(t) for t in study.trials]