import pickle
from typing import Literal
from abc import ABC, abstractmethod
from pathlib import Path
from warnings import warn



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
        Note: the params_distributions and fixed_params must always be optional
        (they must implement defaults).
        
        Parameters:
            preprocessing (Literal["base", "density_filter", "pca"]): 
                Preprocessing strategy to use.
            
            seed (int): Seed for reproducibility.
            
            params_distributions (dict | None, optional):
                Dict of param:distributions from which to sample values in the tuning process.
                Can be None.
            
            fixed_params (dict, optional):
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

    @abstractmethod
    def save(self, filepath: str | Path):
        '''Seriealize the instance using pickle'''
        if not hasattr(self, "estimator_"):
            warn(
                message="The estimator instance is not fitted! Is this expected?", 
                category=UserWarning
            )
        with open(filepath, "wb") as f:
            pickle.dump(self, f)