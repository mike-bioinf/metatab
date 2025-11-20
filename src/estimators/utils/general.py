from typing import Any
from copy import deepcopy
from numpy.random import RandomState



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



def add_string_to_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to add at the beginning of dict keys a string.
    This is helpful when using sklearn pipelines.
    Notes that the function assumes that the keys are str.
    Returns a new dict.
    '''
    return {f"{string}{k}":v for k, v in params_dict.items()}



def remove_string_from_params(params_dict: dict[str, Any], string: str) -> dict:
    '''
    Utility to remove the string from the beginning of params dict keys.
    Notes that the function assumes that the keys are of str type.
    Returns a new dict.
    '''
    new_params_dict = {}
    for key, value in params_dict.items():
        new_key = key.removeprefix(string)
        new_params_dict[new_key] = value
    return new_params_dict



def update_dict( 
    dictionary: dict, 
    name_key: str, 
    value: Any, 
    copy: bool = False
) -> dict:
    '''
    Update the dict or a deepcopy of it with the name_key:value couple.
    Returns the updated dict.
    '''
    dictionary = deepcopy(dictionary) if copy else dictionary
    dictionary[name_key] = value
    return dictionary