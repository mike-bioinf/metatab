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


class PipelineFitError(Exception):
    '''
    Error used to indicate a failure in a pipeline fit proecess.
    '''
    pass