from dataclasses import dataclass


@dataclass
class BagCV:
    '''
    Cross-validated bagging Dataclass.
    Parameters:
        n_repeats (int): Number of cv repeats
        n_folds (int): Number of cv folds
        seed (int): Cv seed
    '''
    n_repeats: int
    n_folds: int
    seed: int