from dataclasses import dataclass


@dataclass
class BagCV:
    '''
    Cross-validation bagging dataclass.
    Parameters:
        n_repeats (int): Number of cv repeats
        n_folds (int): Number of cv folds
        seed (int): Cv seed
    '''
    n_repeats: int
    n_folds: int
    seed: int
    use_oof_as_validation: bool

    @classmethod
    def build_from_dict(cls, dictionary: dict) -> "BagCV":
        return cls(**dictionary)