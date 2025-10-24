import numpy as np
from sklearn.utils.validation import check_is_fitted
from hyperopt.pyll.stochastic import sample



class HyperoptRandomSampler:
    '''Sample randomly from a hyperopt-compatible space'''
    
    def fit(self, space: dict, seed: int) -> "HyperoptRandomSampler":
        '''
        Set the space and seed specifications.
        Parameters:
            space (dict): Hyperopt-compatible space.
            seed (int): Seed used to control the randomness in the sampling procedure.
        '''
        self.space = space
        self.seed = seed
        self.rng_sampling = np.random.default_rng(self.seed)
        self.is_fitted_ = True
        return self
    
    def sample_points(self, n_points: int) -> list[dict]:
        '''
        Get `n_points` from the space.
        Returns the list of sampled points.
        '''
        check_is_fitted(self, "is_fitted_")
        return [sample(self.space, self.rng_sampling) for _ in range(n_points)]