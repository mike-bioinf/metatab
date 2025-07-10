def add_string_to_params(params_dict: dict, string: str):
    '''
    Utility to add at the beginning of the keys of the dict 
    of parametersa string. This is helpful when using sklearn pipelines.
    Returns a new dict.
    '''
    return {f"{string}{k}":v for k, v in params_dict.items()}