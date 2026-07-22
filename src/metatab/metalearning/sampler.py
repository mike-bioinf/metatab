import numpy as np
from sklearn.utils.validation import check_is_fitted
from hyperopt.pyll.stochastic import sample
from hyperopt import rand, fmin



class HyperoptRandomSampler:
    '''
    Sample randomly from a hyperopt-compatible space.

    Parameters:
        follow_hyperopt_fmin (bool, optional):
            Whether to sample "following" the `fmin` hyperopt utility.
            Adds overhead but is quite fast in practice.
            If False the sampling is done via the `sample` utility which
            return different results even when seeded with the same value.
    '''
    def __init__(self,follow_hyperopt_fmin: bool = True):
        self.follow_hyperopt_fmin=follow_hyperopt_fmin


    def fit(self, space: dict, seed: int) -> "HyperoptRandomSampler":
        '''
        Set the space and seed specifications.
        Parameters:
            space (dict): Hyperopt-compatible space.
            seed (int): Seed used to control the randomness in the sampling procedure.
        '''
        self.space = space
        self.seed = seed
        self.is_fitted_ = True
        return self
    

    def sample_points(self, n_points: int) -> list[dict]:
        '''
        Get `n_points` from the space.
        Returns the list of sampled points.
        '''
        check_is_fitted(self, "is_fitted_")
        rng = np.random.default_rng(self.seed)

        if self.follow_hyperopt_fmin:
            points = []

            def mock_eval_func(point):
                points.append(point)
                return 0
            
            _ = fmin(
                fn=mock_eval_func,
                space=self.space,
                algo=rand.suggest,
                max_evals=n_points,
                rstate=rng,
                verbose=False
            )

            return points
        
        else:
            return [sample(self.space, rng) for _ in range(n_points)]