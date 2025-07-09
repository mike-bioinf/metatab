from numpy.random import RandomState
from scipy.stats import loguniform
from copy import deepcopy



def float_to_int(rvs):
    def rvs_wrapper(*args, **kwargs):
        return rvs(*args, **kwargs).round().astype(int)
    return rvs_wrapper


def int_loguniform(low, high):
    '''
    Function to create a loguniform scipy object 
    that returns integers via the "rsv" method.
    '''
    lu = loguniform(low, high)
    lu.rvs = float_to_int(lu.rvs)
    return lu


def get_fresh_random_state(random_state: None | int | RandomState) -> RandomState:
    '''
    Get a fresh random state instance.
    If the input is None it generates a new instance seeded with 0.
    If int then it produces the random state using it as seed.
    If RandomState it returns a deepcopy of it.
    '''
    if random_state is None:
        return RandomState(0)
    elif isinstance(random_state, int):
        return RandomState(random_state)
    elif isinstance(random_state, RandomState):
        return deepcopy(random_state)
    else:
        raise ValueError("Unsupported input.")