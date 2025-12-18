from typing import Any



class ConfigSearchCV:
    '''
    Class that holds the globally configurable settings for SearchCV instances.

    - raise_error_during_search (bool):
        Control whether to ignore the errors during the search.
        If True an error in the fitting process of one point  
        will determine the failure of the entire search.

    - refit_with_best_hps (bool): 
        Control whether SearchCV refit the estimator with the best hps.
    
    - build_df_search (bool): 
        Control whether SearchCV builds the df_search when fitted.
    '''
    raise_error_during_search = False
    refit_with_best_hps = True
    build_df_search = False
    _attrs = [
        "raise_error_during_search",
        "refit_with_best_hps", 
        "build_df_search", 
    ]

    @classmethod
    def get_setting(cls, value: Any, attr: str) -> Any:
        '''
        Returns the global configuration or input value for the specified attribute. 
        The fallback on the global setting happens when the input value is None.
        '''
        if attr not in cls._attrs:
            raise ValueError(f"attr must be one of '{cls._attrs}'")
        if value is None:
            return getattr(cls, attr)
        else:
            return value