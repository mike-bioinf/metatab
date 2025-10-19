import numpy as np
from sklearn.utils.validation import check_is_fitted
from hyperopt.pyll.stochastic import sample
from hp_search.utils import apply_hyperopt_corrections_to_sampled_point



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

    def sample_points(self, n_points: int, apply_hyperopt_corrections: bool = True) -> list[dict]:
        '''
        Get `n_points` from the space.
        Allows to optionally apply a set of hyperopt corrections defined by us to the sampled points.
        Returns the list of sampled points.
        '''
        check_is_fitted(self, "is_fitted_")
        sampled_points = []

        for _ in range(n_points):
            point = sample(self.space, self.rng_sampling)
            if apply_hyperopt_corrections:
                point = apply_hyperopt_corrections_to_sampled_point(point)
            sampled_points.append(point)
     
        return sampled_points
