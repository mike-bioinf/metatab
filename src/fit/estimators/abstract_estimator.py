from typing import Literal
from abc import ABC, abstractmethod



class AbstractEstimator(ABC):
    '''
    Blueprint for estimators classes.
    The estimators classes must set the 'estimator_' attribute
    in the "fit" method, storing the model fitted on the input data.
    '''
    @abstractmethod
    def __init__(
        self, 
        preprocessing: Literal["base", "density_filter", "pca"],
        seed: int,
        params_distributions: dict | None,
        fixed_params: dict
    ):
        '''
        Parameters:
            preprocessing (Literal["base", "density_filter", "pca"]): 
                Preprocessing strategy to use.
            
            seed (int): Seed for reproducibility.
            
            params_distributions (dict | None):
                Dict of param:distributions from which to sample values in the tuning process.
                Can be None.
            
            fixed_params (dict):
                Dict of param:value that are fixed i.e. not tuned in the search.
        '''
        self.preprocessing = preprocessing
        self.seed = seed
        self.params_distributions = params_distributions
        self.fixed_params = fixed_params
        

    @abstractmethod
    def fit(*args, **kwargs):
        pass

    @abstractmethod
    def predict_proba(*args, **kwargs):
        pass