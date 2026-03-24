class DeviceError(Exception):
    '''
    Error used to indicate incompatible or not available device.
    '''
    pass


class TimeLimitError(Exception):
    '''
    Error used to indicate time limit violations.
    '''
    pass


## REFACTOR: delete this?
class DFSearchBuildingError(Exception):
    '''
    Error used to indicate problems in the bulding process of the dataframe summarizing the search.
    '''
    pass


class PipelineFitError(Exception):
    '''
    Error used to indicate a failure in a pipeline fit proecess.
    '''